# Gaia Benchmarking
Benchmarking scripts for Gaia

## Installation

```bash
git clone https://github.com/yourusername/gaia-benchmark.git
cd gaia-benchmark
pip install -r requirements.txt
```

## Benchmark Directories

### `benchmark_sequence/`
Sequence similarity search benchmark on the [OG_prot90 dataset](https://huggingface.co/datasets/tattabio/OG_prot90). Uses BLASTp results as ground truth to evaluate recall@k performance.

### `benchmark_context/`
Genomic context retrieval sensitivity benchmark. Recall is calculated based on the retrieval of genes with similar genomic context (proteins in context matching at >50% sequence identity and >50% sequence coverage) within the top K retrievals. Uses the [OG_prot90 dataset](https://huggingface.co/datasets/tattabio/OG_prot90).

### `benchmark_structure/`
Protein structure similarity search benchmark. Evaluates retrieval of proteins with similar structures using the [SCOPe-40 test dataset](https://scop.berkeley.edu/).

### `benchmark_bac_arch/`
Benchmark for remote homology matching between functional homologs of bacterial (*E. coli* K-12) and archaeal (*S. acidocaldarius* DSM 639) proteins. Uses the [bac_arch_bigene](https://huggingface.co/datasets/tattabio/bac_arch_bigene) dataset from [DGEB](https://github.com/TattaBio/dgeb)


## Dataset preparation

### `prepare_data/`
Scripts for sequence embedding and setting up vector search with Qdrant.

## Citation

```bibtex
@article{jha2024gaia,
  title={Gaia: An AI-enabled Genomic Context-Aware Platform for Protein Sequence Annotation},
  author={Jha, Nishant and Kravitz, Joshua and West-Roberts, Jacob and Camargo, Antonio and Roux, Simon and Cornman, Andre and Hwang, Yunha},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2024.11.19.624387}
}
```



