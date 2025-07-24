from datasets import load_dataset

def load_streaming_dataset(files, text_column, tokenizer):
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    def tokenize_fn(batch):
        return tokenizer(batch[text_column], add_special_tokens=False)
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds.features.keys() if c != text_column])
    return tokenized