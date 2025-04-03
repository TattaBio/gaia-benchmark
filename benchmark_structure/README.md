## Sequence Retrieval Benchmark

The SCOPe40-test dataset is used to benchmark methods on retrieval of similar structures.
Retrieval results for BLASTp, MMseqs2, and Foldseek are copied from https://zenodo.org/records/11480660

### 1. Run ESM2 retrieval

```bash
python search_esm2.py
```

### 2. Run ESM2 retrieval

```bash
python search_glm2.py
```

### 2. Plot Results
See `plot_results.ipynb`, which uses the results from `search_result` directory for each method.