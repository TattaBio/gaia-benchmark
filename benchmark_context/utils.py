import datasets
import subprocess
import tempfile
from typing import List
import polars as pl
from qdrant_client import QdrantClient, models
from collections import namedtuple

# Define named tuple for sequence information
SequenceInfo = namedtuple('SequenceInfo', ['sequence', 'id', 'is_match'])

def find_neighboring_cds(cds_ids_df: pl.DataFrame, ds: datasets.Dataset, query_cds_id: str, n_neighbors: int = 5):
    """
    Find n_neighbors CDS entries before and after a given CDS ID, including both IDs and sequences.
    
    Args:
        cds_ids_df: Polars DataFrame containing CDS information (must be exploded with row_idx)
        ds: HuggingFace dataset containing both CDS IDs and sequences
        query_cds_id: The CDS ID to find neighbors for
        n_neighbors: Number of neighbors to find before and after (default: 5)
    
    Returns:
        Tuple containing:
        - List of SequenceInfo named tuples containing sequence and ID information
        - Index of the query CDS in the returned list
    """
    # Find the row index
    query_row_idx = cds_ids_df.filter(pl.col('CDS_ids') == query_cds_id).select('row_idx').item()

    # Get the corresponding row from the HuggingFace dataset
    ds_row = ds[query_row_idx]
    
    cds_ids = ds_row['CDS_ids']
    cds_seqs = ds_row['CDS_seqs']
    
    # Find the index of the query CDS
    query_idx = cds_ids.index(query_cds_id)
    
    # Get the sequences and IDs for neighbors
    start_idx = max(0, query_idx - n_neighbors)
    end_idx = min(len(cds_ids), query_idx + n_neighbors + 1)
    
    # Create the contig list with all sequences using list comprehension
    sequences = [
        SequenceInfo(
            sequence=cds_seqs[i], 
            id=cds_ids[i],
            is_match=(i == query_idx)
        ) for i in range(start_idx, end_idx)
    ]
    return sequences

def context_search(
    cds_id: str,
    client: QdrantClient,
    collection_name: str,
    search_limit: int,
    cds_ids_df: pl.DataFrame,
    ds: datasets.Dataset,
) -> list[dict]:
    """Search for similar sequences and get their genomic context.
    
    Args:
        cds_id: ID of the query CDS sequence
        client: Qdrant client instance
        collection_name: Name of Qdrant collection to search
        search_limit: Maximum number of search results to return
        cds_ids_df: DataFrame containing CDS IDs and row indices
        ds: Dataset containing CDS sequences and metadata
        
    Returns:
        List of dictionaries containing genomic context for each search result
    """
    (records, _) = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="cds_id", match=models.MatchAny(any=[cds_id]))]
        ),
        with_vectors=True,
        with_payload=True,
        limit=1,
    )
    assert len(records) == 1
    vector = records[0].vector

    results = client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    with_payload=True,
                    limit=search_limit,
                )
    search_result_ids = [r.payload["cds_id"] for r in results]
    
    contexts = [find_neighboring_cds(cds_ids_df, ds, retrieved_id) for retrieved_id in search_result_ids]
    return contexts


def calculate_context_recall(responses: List[List[SequenceInfo]]) -> List[List[int]]:
    """Calculate context recall for a list of responses using BLAST.
    
    Args:
        responses: List of lists of SequenceInfo named tuples, where each inner list represents
                  a contig of sequences from a search result.
    
    Returns:
        List of lists containing the number of matching contexts for each search result.
        Each inner list contains the number of matching contexts for each position in the search results.
    """
    all_results = []
    for response in responses:            
        # Use first sequence as query context
        best_match = response[0]
        query_context = [seq.sequence for seq in best_match if not seq.is_match]
        
        # Get contexts from all other results
        retrieved_contexts = []
        for match in response[1:]:
            retrieved_contexts.append([seq.sequence for seq in match if not seq.is_match])
        
        # Run BLAST to find matching contexts
        results, _ = run_blastp(query_context, retrieved_contexts)
        all_results.append(results)
    return all_results


def run_blastp(queries: List[str], contexts: List[List[str]]) -> List[int]:
    # If no contexts, return zeros
    if not contexts:
        return [0 for _ in range(len(contexts))], []
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as query_file, \
         tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as subject_file, \
         tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as output_file:

        # Write query to file
        for i, query in enumerate(queries):
            query_file.write(f">{i}\n{query}\n")

        for i, context in enumerate(contexts):
            for j, subject in enumerate(context):
                subject_file.write(f">{i}_{j}\n{subject}\n")

        # Close files to ensure all data is written
        query_file.close()
        subject_file.close()
        output_file.close()
        # Run BLAST
        try:
            print("Creating BLAST database")
            subprocess.run(
                ["makeblastdb", "-in", subject_file.name, "-dbtype", "prot"],
                check=True,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error running makeblastdb: {e.stderr}")
        print("Running BLASTp")
        try:
            subprocess.run(
                [
                    "blastp",
                    "-query",
                    query_file.name,
                    "-db",
                    subject_file.name,
                    "-out",
                    output_file.name,
                    "-logfile",
                    "/dev/null",
                    "-outfmt",
                    '6 sseqid qseqid pident qseq sseq qcovs',
                    "-num_threads",
                    "6",
                    "-max_target_seqs",
                    "1000",  # Ensure we get all hits
                ],
                check=True,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error running blastp: {e.stderr}")

        # Parse BLAST output
        correct_context = [0 for _ in range(len(contexts))]
        correct_context_info = [[] for _ in range(len(contexts))]
        with open(output_file.name, "r") as f:
            for line in f:
                sseqid, qseqid, pident, qseq, sseq, qcovs = line.strip().split("\t")
                result_index = int(sseqid.split("_")[0])
                if float(pident) > 50 and int(qcovs) > 50:
                    correct_context[result_index] += 1
                    correct_context_info[result_index].append((qseqid, sseqid, pident, qseq, sseq, qcovs))
        print(correct_context)
        return correct_context, correct_context_info

