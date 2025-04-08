"""Tokenize CDS dataset and save to disk."""
import argparse
import os
from os.path import join
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_function(ids, sequences, tokenizer, max_seq_length: int):
    if 'gLM2' in tokenizer.__class__.__name__:
        # Add the direction token if using a gLM2 tokenizer.
        sequences = [f"<+>{seq}" for seq in sequences]
    tokenized_data = tokenizer(
        sequences, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='np')
    output = {
        'input_ids': tokenized_data['input_ids'].astype(np.int16),
        'attention_mask': tokenized_data['attention_mask'].astype(bool),
        'id': ids,
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='tattabio/OG_prot90',
        help="Huggingface dataset name.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='facebook/esm2_t6_8M_UR50D',
        help="Huggingface tokenizer name.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length. Longer sequences will be truncated.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The path to write tokenized dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=20,
        help="Number of processes to use for tokenization.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    # Create the cache dir and write the dataset config to yaml.
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"*** Load Dataset {args.dataset_name} ***")
    ds = load_dataset(args.dataset_name, cache_dir=args.save_dir)

    tokenize_function_kwargs = {
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
    }

    print("*** Dataset Tokenize ***")
    tokenized_ds = ds.map(
        tokenize_function,
        input_columns=['id', 'sequence'],
        batched=True,
        num_proc=args.num_proc,
        fn_kwargs=tokenize_function_kwargs,
        remove_columns=ds.column_names
    )

    print("*** Dataset Set Torch Format ***")
    tokenized_ds = tokenized_ds.with_format(type="torch")

    print("*** Save dataset to disk ***")
    tokenized_ds_path = join(args.save_dir, 'tokenized_ds')
    tokenized_ds.save_to_disk(tokenized_ds_path, num_proc=args.num_proc)

    print(f"Saved to: {tokenized_ds_path}")

    num_cleaned = ds.cleanup_cache_files()
    print(f"Cleaned {num_cleaned} cache files.")


if __name__ == "__main__":
    main()