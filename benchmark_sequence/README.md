## Sequence Retrieval Benchmark

### 1. Sample Query Proteins
See [prepare_data](/prepare_data) directory for generating `OG_90.fasta` file.

```bash
seqkit sample -n 1200 OG_90.fasta > queries.fasta
```

### 2. BLASTp Ground Truth Generation
```bash
python create_blastp_groundtruth.py \
    --query_fasta queries.fasta \
    --target_fasta OG_90.fasta \
    --output_dir blastp_output_dir \
    --threads 20
```

### 3. MMseqs2 Search

1. Create MMseqs2 database:
```bash
mmseqs createdb OG_90.fasta OG_90_mmseqs_db
```

2. Run MMseqs2 search:
```bash
mmseqs easy-search queries.fasta OG_90_mmseqs_db data/mmseqs_results_seq.m8 tmp --db-load-mode 2 --threads 20
```

### 4. Qdrant Evaluation (for gLM2 and ESM2)

See [prepare_data](/prepare_data) directory for generating Qdrant collections. Then run evaluation:

```bash
python qdrant_eval.py \
    http://localhost:6333 gLM2_650M_embed_collection \
    -f data/blastp_groundtruth_results.tsv -k 100
```

```bash
python qdrant_eval.py \
    http://localhost:6333 ESM2_650M_collection \
    -f data/blastp_groundtruth_results.tsv -k 100
```

### 5. Plot Results
See `plot_results.ipynb`, which uses the results from `data` directory for each method.