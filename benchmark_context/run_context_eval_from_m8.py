import argparse
import datasets
import pickle
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd

from utils import calculate_context_recall, find_neighboring_cds_batch

TOP_K = 10
HF_DATASET_NAME = "tattabio/OG"

def parse_mmseqs_results(m8_file: str, top_k: int = TOP_K):
    """Parse MMseqs search results from m8 format."""
    # Read m8 file into DataFrame
    # Format: query, target, identity, alignment_length, mismatches, gap_opens, q_start, q_end, t_start, t_end, evalue, bitscore
    df = pd.read_csv(m8_file, sep='\t', header=None,
                     names=['query', 'target', 'identity', 'aln_len', 'mismatches', 
                           'gap_opens', 'q_start', 'q_end', 't_start', 't_end', 
                           'evalue', 'bitscore'])
    
    # Group by query and get first k results for each
    results = {}
    for query, group in df.groupby('query'):
        # Get first k results
        results[query] = group['target'].head(top_k + 1).tolist()
    
    return results

def main(m8_file: str, save_name: str, test_file: str):
    # Load data
    ds = datasets.load_dataset(HF_DATASET_NAME, cache_dir='/mnt/raid0/datasets/hf/OG_gLM2_2048')['train']
    cds_df = ds.remove_columns([col for col in ds.column_names if col not in ['CDS_ids', 'CDS_seqs']]).to_polars()
    cds_ids_df = cds_df.select('CDS_ids').with_row_index('row_idx').explode('CDS_ids')
    
    # Get test sequence IDs
    ids = [record.id for record in SeqIO.parse(test_file, "fasta")]
    
    # Parse MMseqs results
    mmseqs_results = parse_mmseqs_results(m8_file)
    
    # Run evaluation
    responses = []
    for cds_id in tqdm(ids, total=len(ids)):            
        # Get top k matches from MMseqs results
        matches = mmseqs_results[cds_id]
        
        # Process all matches for this query at once
        contexts = find_neighboring_cds_batch(cds_ids_df, cds_df, matches)
        responses.append(contexts)
    
    results = calculate_context_recall(responses)
    
    # Save results
    with open(save_name, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_name}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m8_file", required=True, help="Path to MMseqs search results in m8 format")
    parser.add_argument("--save_name", required=True, help="Name of the file to save results to")
    parser.add_argument("--test_file", required=True, help="Path to test sequences fasta file")
    args = parser.parse_args()
    
    main(args.m8_file, args.save_name, args.test_file) 