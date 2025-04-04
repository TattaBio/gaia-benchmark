import argparse
import datasets
import pickle
from qdrant_client import QdrantClient
from Bio import SeqIO
from tqdm import tqdm

from utils import context_search, calculate_context_recall

TOP_K = 10
HF_DATASET_NAME = "tattabio/OG"

def main(collection_name: str, save_name: str, test_file: str, qdrant_api: str):
    # Initialize client and load data
    client = QdrantClient(url=qdrant_api, timeout=1000, prefer_grpc=False)
    ds = datasets.load_dataset(HF_DATASET_NAME)['train']
    cds_ids_df = ds.remove_columns([col for col in ds.column_names if col != 'CDS_ids']).to_polars().with_row_index('row_idx').explode('CDS_ids')
    
    ids = [record.id for record in SeqIO.parse(test_file, "fasta")]
    
    # Run evaluation
    responses = []
    for cds_id in tqdm(ids, total=len(ids)):
        response = context_search(
            cds_id=cds_id,
            client=client,
            collection_name=collection_name,
            search_limit=TOP_K+1,
            cds_ids_df=cds_ids_df,
            ds=ds
        )
        responses.append(response)
    
    results = calculate_context_recall(responses)
    
    # Save results
    with open(save_name, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_name}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection")
    parser.add_argument("--save_name", required=True, help="Name of the file to save results to")
    parser.add_argument("--test_file", required=True, help="Path to test sequences fasta file")
    parser.add_argument("--qdrant_api", required=True, help="Qdrant API URL")
    args = parser.parse_args()
    
    main(args.collection, args.save_name, args.test_file, args.qdrant_api)