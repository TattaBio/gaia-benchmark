# Data preparation
Scripts for preparing data for Gaia benchmarking including dataset embedding and setting up Qdrant vector database.
This is only needed for running `benchmark_sequence` and `benchmark_context`.

## Convert HuggingFace dataset to FASTA
Convert a HuggingFace dataset to FASTA format for use with sequence search tools (BLASTp and MMSeqs):

```bash
python hf_to_fasta.py \
    --dataset_name=tattabio/OG \
    --output_file=OG_prot90.fasta
```

## Tokenize dataset
First tokenize and save the huggingface dataset.
```bash
python tokenize_dataset.py  \
  --dataset_name=tattabio/OG \
  --tokenizer_name=tattabio/gLM2_650M_embed  \
  --max_seq_length=2048  \
  --save_dir=OG_gLM2 \
  --num_proc=20
```

Or for ESM2 preprocessing:
```bash
python tokenize_dataset.py  \
  --dataset_name=tattabio/OG \
  --tokenizer_name=facebook/esm2_t33_650M_UR50D  \
  --max_seq_length=1024  \
  --save_dir=OG_ESM2  \
  --num_proc=20
```

## Embed proteins

Run inference and save protein embeddings and ids to HDF5 file.

With gLM2 embedding model:

```bash
accelerate launch --config_file configs_accelerate/ddp.yaml \
    embed_glm2.py \
    --dataset_path=OG_gLM2/tokenized_ds \
    --output_dir=OG_gLM2/embedding \
    --model_name="tattabio/gLM2_650M_embed" \
    --batch_size=128 \
    --save_ids=True
```


Or with ESM2
```bash
accelerate launch --config_file configs_accelerate/ddp.yaml --mixed_precision=fp16  \
    embed_esm.py \
    --dataset_path=OG_ESM2/tokenized_ds \
    --output_dir=OG_gLM2/embedding \
    --model_name="facebook/esm2_t33_650M_UR50D" \
    --batch_size=128 \
    --layers=mid \
    --save_ids=True
```

## Upload to Qdrant

Install and run Qdrant Vector Database using Docker:

```bash
docker pull qdrant/qdrant
```

```bash
docker run -p 6333:6333 \
    -v $(pwd)/path/to/data:/qdrant/storage \
    qdrant/qdrant
```

Upload the embeddings to a Qdrant collection.

For gLM2 embeddings:
```bash
cd qdrant_upload
cargo run -- \
    --collection-name=gLM2_650M_embed_collection \
    --embeddings-file=/path/to/gLM2/embeddings.h5 \
    --embedding-dim=512 \
    --qdrant-url=http://localhost:6333
```

For ESM2 embeddings:
```bash
cd qdrant_upload
cargo run -- \
    --collection-name=ESM2_650M_collection \
    --embeddings-file=/path/to/ESM2/embeddings.h5 \
    --embedding-dim=1280 \
    --qdrant-url=http://localhost:6333
```
