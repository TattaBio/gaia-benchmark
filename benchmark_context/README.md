# Context Retrieval Benchmark

Genomic context retrieval sensitivity benchmark comparing Gaia search and ESM2 embedding-based search, where recall is calculated based on the presence of a gene with a similar genomic context (at least 70% of the proteins in context matching at >50% sequence identity and >50% sequence coverage) within the top K retrievals. We benchmark these methods against MMseqs2 and BLASTp.

## Run gLM2 and ESM2 context retrieval.

For gLM2:
```bash
python run_context_eval.py \
    --collection "gLM2_650M_embed_collection" \
    --save_name "glm2_results_3k.pkl" \
    --test_file "data/cluster_sampled_3k.fasta" \
    --qdrant_api "http://localhost:6333"
```

For ESM2:
```bash
python run_context_eval.py \
    --collection "ESM2_650M_collection" \
    --save_name "esm2_results_3k.pkl" \
    --test_file "data/cluster_sampled_3k.fasta" \
    --qdrant_api "http://localhost:6333"
```


## MMseqs Search Evaluation

Evaluates context sensitivity using sequence search through MMseqs:

```bash
mmseqs easy-search data/cluster_sampled_3k.fasta mmseqs_90_db data/cluster_sampled_3k_mmseqs_result.m8 tmp --db-load-mode 2 --threads 20
```

```bash
python run_context_eval_from_m8.py \
    --m8_file "data/cluster_sampled_3k_mmseqs_result.m8" \
    --save_name "mmseqs_results_3k.pkl" \
    --test_file "data/cluster_sampled_3k.fasta"
```

## BLASTp Search Evaluation

Evaluates context sensitivity using sequence search through BLASTp:

```bash
blastp -query data/cluster_sampled_3k.fasta -db blastdb -out data/cluster_sampled_3k_blast_result.m8 -outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore' -num_threads 20
```

```bash
python run_context_eval_from_m8.py \
    --m8_file "data/cluster_sampled_3k_blast_result.m8" \
    --save_name "blastp_results_3k.pkl" \
    --test_file "data/cluster_sampled_3k.fasta"
```