import random
import torch
from torch.utils.data import IterableDataset

class ConstantLengthDataset(IterableDataset):
    """
    Streams tokenized samples and concatenates them into constant-length chunks.
    Assumes each element is a dict with key 'input_ids'.
    """
    def __init__(self, tokenized_stream, seq_length, eos_token_id, shuffle_buffer=1_000_000):
        self.stream = tokenized_stream
        self.seq_length = seq_length
        self.eos = eos_token_id
        self.shuffle_buffer = shuffle_buffer
        self.buffer = []
        self.generator = random.Random(42)

    def __iter__(self):
        token_buffer = []
        for sample in self._shuffled_stream():
            ids = sample["input_ids"]
            token_buffer.extend(ids + [self.eos])
            while len(token_buffer) >= self.seq_length:
                chunk = token_buffer[: self.seq_length]
                token_buffer = token_buffer[self.seq_length:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(len(chunk), dtype=torch.long),
                }

    def _shuffled_stream(self):
        # Reservoir-like shuffle for streaming datasets
        for example in self.stream:
            if self.shuffle_buffer > 0:
                if len(self.buffer) < self.shuffle_buffer:
                    self.buffer.append(example)
                    continue
                # swap
                idx = self.generator.randint(0, len(self.buffer) - 1)
                yield self.buffer[idx]
                self.buffer[idx] = example
            else:
                yield example
        # flush remaining
        self.generator.shuffle(self.buffer)
        for ex in self.buffer:
            yield ex
        self.buffer = []