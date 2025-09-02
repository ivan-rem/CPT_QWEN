# constant_length_dataset.py
import random
from typing import Any, Dict, Iterable, Optional, Union, List

import torch
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist

try:
    from transformers import PreTrainedTokenizerBase  # type: ignore
except Exception:  # pragma: no cover
    PreTrainedTokenizerBase = object


class ConstantLengthDataset(IterableDataset):
    """
    Packs a stream of examples into fixed-length chunks for causal LM pretraining.

    Key points:
      - Accepts pre-tokenized samples with 'input_ids' OR raw text via `text_column`.
      - Tokenizes long raw text in fixed-size windows (no giant sequences).
      - Shards across DDP ranks and per DataLoader worker.
      - Reservoir shuffle at the example level.
      - Emits dicts with tensors: input_ids, labels, attention_mask (all length `seq_length`).
    """

    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        seq_length: int,
        eos_token_id: int,
        shuffle_buffer: int = 1_000_000,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        text_column: str = "text",
        append_eos: bool = True,
        # Max tokens per tokenizer call when processing raw text.
        # Default = seq_length so the tokenizer never emits >seq_length in one shot.
        tokenization_chunk_tokens: Optional[int] = None,
    ):
        # DDP rank/world
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # Rank shard if supported (HF Datasets)
        self._base_dataset = dataset
        if hasattr(self._base_dataset, "shard"):
            try:
                self._base_dataset = self._base_dataset.shard(
                    num_shards=self.world_size, index=self.rank
                )
            except TypeError:
                self._base_dataset = self._base_dataset.shard(self.world_size, self.rank)

        self.seq_length = int(seq_length)
        self.eos = int(eos_token_id)
        self.shuffle_buffer = int(shuffle_buffer)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.append_eos = bool(append_eos)
        self.tokenization_chunk_tokens = int(tokenization_chunk_tokens or self.seq_length)

        # Reservoir buffer + RNG
        self._reservoir: List[Dict[str, Any]] = []
        base_seed = 42 + 997 * self.rank
        self._rng = random.Random(base_seed)

    def _worker_sharded_dataset(self):
        ds = self._base_dataset
        wi = get_worker_info()
        if wi is not None and hasattr(ds, "shard") and wi.num_workers and wi.num_workers > 1:
            try:
                ds = ds.shard(num_shards=wi.num_workers, index=wi.id)
            except TypeError:
                ds = ds.shard(wi.num_workers, wi.id)
        return ds

    def _token_windows_from_text(self, txt: str) -> List[List[int]]:
        """
        Tokenize raw text into <= tokenization_chunk_tokens windows using the tokenizer's
        overflow mechanism so we never create giant sequences (and avoid warnings).
        """
        if not txt:
            return []
        if self.tokenizer is None:
            return []

        enc = self.tokenizer(
            txt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.tokenization_chunk_tokens,
            return_overflowing_tokens=True,
            return_attention_mask=False,
        )

        # Robustly gather windows from both fast/slow tokenizers
        windows: List[List[int]] = []
        # Prefer enc.encodings when available (fast tokenizers)
        encodings = getattr(enc, "encodings", None)
        if encodings:
            for e in encodings:
                ids = getattr(e, "ids", None)
                if ids:
                    windows.append(list(ids))
        else:
            ids_field = enc.get("input_ids", [])
            if isinstance(ids_field, list) and ids_field:
                if isinstance(ids_field[0], list):  # list of windows
                    for w in ids_field:
                        if w:
                            windows.append(list(w))
                else:
                    windows.append(list(ids_field))
        return windows

    def _example_to_token_windows(self, ex: Dict[str, Any]) -> List[List[int]]:
        """
        Return a list of token windows (each a List[int]) for a single example.
        - If 'input_ids' present -> single window with those ids.
        - Else tokenize raw text into multiple windows.
        """
        # Pre-tokenized path
        ids = ex.get("input_ids", None)
        if ids is not None:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            elif isinstance(ids, (list, tuple)):
                ids = list(ids)
            else:
                return []
            return [ids] if ids else []

        # Raw text path
        txt = ex.get(self.text_column, None)
        if not isinstance(txt, str):
            return []
        return self._token_windows_from_text(txt)

    def _shuffled_examples(self, ds: Iterable[Dict[str, Any]]):
        if self.shuffle_buffer <= 0:
            for ex in ds:
                yield ex
            return

        buf = self._reservoir
        buf.clear()
        for ex in ds:
            if len(buf) < self.shuffle_buffer:
                buf.append(ex)
                continue
            idx = self._rng.randint(0, len(buf) - 1)
            yield buf[idx]
            buf[idx] = ex

        self._rng.shuffle(buf)
        for ex in buf:
            yield ex
        buf.clear()

    def __iter__(self):
        # Reseed per worker
        wi = get_worker_info()
        if wi is not None:
            worker_seed = (self._rng.randint(0, 2**31 - 1) ^ (wi.id * 104729 + 1)) & 0x7FFFFFFF
            self._rng.seed(worker_seed)

        ds = self._worker_sharded_dataset()
        token_buffer: List[int] = []

        for ex in self._shuffled_examples(ds):
            windows = self._example_to_token_windows(ex)
            if not windows:
                continue

            for toks in windows:
                if not toks:
                    continue
                if self.append_eos:
                    token_buffer.extend(toks + [self.eos])
                else:
                    token_buffer.extend(toks)

                while len(token_buffer) >= self.seq_length:
                    chunk = token_buffer[: self.seq_length]
                    token_buffer = token_buffer[self.seq_length:]

                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone(),
                        "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
                    }
        # Drop remainder (tail) by design for CPT
