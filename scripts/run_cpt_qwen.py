#!/usr/bin/env python
import os
import math
from dataclasses import dataclass, field
from typing import Optional, List
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

from src.data.constant_length_dataset import ConstantLengthDataset
from src.utils.io import load_streaming_dataset
from src.utils.wandb_callbacks import LogTrainTokensCallback

# ------------------------
# Dataclass args 
# ------------------------
@dataclass
class ModelArguments:
    model_name: str = field(default="Qwen/Qwen2.5-3B")

@dataclass
class DataArguments:
    train_files: str = field(default=None)
    validation_files: Optional[str] = None
    text_column: str = field(default="text")
    max_seq_len: int = field(default=4096)
    pack_sequences: bool = field(default=True)
    shuffle_buffer_size: int = field(default=1_000_000)

@dataclass
class ExtraArguments:
    flash_attention_2: bool = field(default=True)
    use_deepspeed: bool = field(default=False)
    zero_stage: int = field(default=2)
    wandb_project: str = field(default="qwen-cpt")
    wandb_run_name: str = field(default="qwen2.5-3b_cpt_run")
    resume_checkpoint: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume from checkpoint if available."}
    )
    num_train_steps: Optional[int] = field(default=None, metadata={"help": "Number of training steps"})


# ------------------------
# Helpers
# ------------------------

def get_datasets(tokenizer, dargs: DataArguments):
    # Streaming tokenize
    train_stream = load_streaming_dataset(dargs.train_files, dargs.text_column, tokenizer)
    if dargs.pack_sequences:
        train_ds = ConstantLengthDataset(train_stream, dargs.max_seq_len, tokenizer.eos_token_id, dargs.shuffle_buffer_size)
    else:
        train_ds = train_stream

    eval_ds = None
    if dargs.validation_files:
        eval_stream = load_streaming_dataset(dargs.validation_files, dargs.text_column, tokenizer)
        if dargs.pack_sequences:
            eval_ds = ConstantLengthDataset(eval_stream, dargs.max_seq_len, tokenizer.eos_token_id, dargs.shuffle_buffer_size // 10)
        else:
            eval_ds = eval_stream
    return train_ds, eval_ds


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExtraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yml", ".yaml")):
        model_args, data_args, training_args, extra_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # WandB setup
    if "wandb" in training_args.report_to:
        os.environ.setdefault("WANDB_PROJECT", extra_args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", extra_args.wandb_run_name)

    # Auto-resume logic
    last_ckpt = None
    if extra_args.resume_checkpoint and os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None:
            print(f"[INFO] Resuming from checkpoint: {last_ckpt}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if extra_args.flash_attention_2 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=attn_impl,
    )

    # Data
    train_ds, eval_ds = get_datasets(tokenizer, data_args)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[LogTrainTokensCallback(data_args.max_seq_len)],
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()

    # Final eval perplexity if eval set exists
    if eval_ds is not None:
        metrics = trainer.evaluate()
        ppl = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss") is not None else None
        if ppl:
            print(f"Final Eval Perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()