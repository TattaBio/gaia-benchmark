"""
Compute recall@k using the best BlastP match as the ground truth.

Recall is calculated by determining if the ground-truth (best BlastP match) is present among the top-k
results retrieved by the vector database.

Example usage:

python qdrant_eval.py \
        http://localhost:6333 gLM2_650M_embed_collection \
            -f data/blastp_groundtruth_results.tsv -k 10
"""

import argparse
import asyncio
import csv
import datetime
import json
import os
import traceback
from datetime import timezone

from pydantic import BaseModel, Field, field_validator
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import SearchParams
from tqdm import tqdm


class ExpectedMatchTestCase(BaseModel):
    query_cds_id: str = Field(alias="query_sequence_id")
    match_cds_ids: list[str] = Field(alias="matched_sequence_ids")
    alignment_scores: list[float] = Field(alias="alignment_identities")

    @field_validator("match_cds_ids", mode="before")
    @classmethod
    def parse_ids(cls, ids_str: str):
        return ids_str.split(",")

    @field_validator("alignment_scores", mode="before")
    @classmethod
    def parse_scores(cls, scores_str: str):
        scores = scores_str.split(",")
        return [float(score) for score in scores]

async def find_by_id_batch(
    client: AsyncQdrantClient,
    protein_ids: list[str],
    collection_name: str,
    id_name: str = "cds_id",
):
    if len(protein_ids) == 0:
        return []

    unique_protein_ids = list(set(protein_ids))
    (records, _) = await client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key=id_name, match=models.MatchAny(any=unique_protein_ids))]
        ),
        with_vectors=True,
        with_payload=True,
        limit=len(protein_ids),
    )

    assert len(records) == len(unique_protein_ids), f"""Found too few or too many records for the provided protein IDs.
           Found {len(records)}, expected {len(unique_protein_ids)}."""

    # Qdrant returns records in arbitrary order, so we need to reorder them
    # to match the order of the user input, and account for possible duplicate ids.
    id_to_record_index = {record.payload['cds_id']: index for index, record in enumerate(records)}
    return [records[id_to_record_index[cds_id]] for cds_id in protein_ids]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qdrant_url",
        help="URL of the Qdrant server",
    )
    parser.add_argument(
        "qdrant_collection",
        help="Name of the Qdrant collection",
    )
    parser.add_argument("--api-key", default=None, type=str, help="QDrant API key (read-only)")
    parser.add_argument("-k", "--top-k", default=10, type=int, help="Number of results to retrieve")
    parser.add_argument(
        "--start",
        default=None,
        type=int,
        help="Index of the first sequence to test (inclusive)",
    )
    parser.add_argument(
        "--end",
        default=None,
        type=int,
        help="Index of the last sequence to test (exclusive)",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to the test set CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the CSV file to save results.",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument("--id", default=None, type=str, help="ID of the CDS to test")
    parser.add_argument(
        "--ef", 
        "--ef-search",
        dest="ef_search",
        default=None, 
        type=int, 
        help="HNSW ef_search parameter (higher values improve recall at cost of speed)"
    )
    args = parser.parse_args()
    tests: list[ExpectedMatchTestCase] = []
    with open(args.file) as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            tests.append(ExpectedMatchTestCase.model_validate(row))

    client = AsyncQdrantClient(url=args.qdrant_url, api_key=args.api_key)
    if not args.silent:
        print(f"Running on collection {args.qdrant_collection} at URL {args.qdrant_url}")

    if (args.start is not None or args.end is not None) and args.id is not None:
        raise ValueError("Cannot specify both --id and --start/--end")

    if args.start is None:
        args.start = 0
    if args.end is None:
        args.end = len(tests)

    args.start = max(0, min(len(tests), args.start))
    args.end = max(0, min(len(tests), args.end))
    tests = tests[args.start : args.end]

    if args.id is not None:
        tests = [t for t in tests if t.query_cds_id == args.id]

    results_path = args.output
    if results_path is None:
        timestamp = datetime.datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        input_filename = os.path.splitext(os.path.basename(args.file))[0]

        results_path = os.path.join(
            os.path.dirname(args.file),
            f"results__{input_filename}__k{args.top_k}__{args.qdrant_collection}__{timestamp}.csv",
        )
    results_file = open(results_path, "w")
    results_writer = csv.writer(results_file)
    results_writer.writerow(
        [
            "query",
            "recall",
            "expected",
            f"search_result (k={args.top_k})",
            "n_search_result",
            "error",
        ]
    )

    all_recalls = []
    for test in tqdm(tests):
        records = await find_by_id_batch(
            client,
            [test.query_cds_id],
            args.qdrant_collection,
        )
        if not records or len(records) == 0:
            msg = f"Could not find protein with id {test.query_cds_id}. Are you sure it is a centroid?"
            if not args.silent:
                tqdm.write(msg)
                tqdm.write("Skipping...")
            results_writer.writerow([test.query_cds_id, "", "", "", "", msg])
            continue

        try:
            query_protein = records[0]
            search_params = None
            if args.ef_search is not None:
                search_params = SearchParams(hnsw_ef=args.ef_search, exact=False)
                
            results = await client.search(
                collection_name=args.qdrant_collection,
                query_vector=query_protein.vector,
                with_payload=True,
                limit=args.top_k,
                search_params=search_params,
            )
            if not args.silent:
                tqdm.write(f"Found {len(results)} results for {test.query_cds_id}")
        except Exception as e:  # noqa: BLE001
            msg = f"Error querying Qdrant: {e}"
            if not args.silent:
                tqdm.write(msg)
                tqdm.write("Skipping...")

            results_writer.writerow(
                [
                    test.query_cds_id,
                    "",
                    "",
                    "",
                    "",
                    msg + f"\n\nTraceback:\n{traceback.format_exc()}",
                ]
            )
            continue

        query_id = test.query_cds_id

        # NOTE: BlastP matches are reported with decreasing bit score.
        # The expected id is the best match, ie. highest bit score.
        expected_id = test.match_cds_ids[0]
        assert all(r.payload is not None for r in results), "Some results are missing payloads"

        # ignore type error since we assert that payloads are not None above
        search_result_ids = [r.payload["cds_id"] for r in results]  # type: ignore
        num_search_result = len(search_result_ids)
        # Recall is 1 if the expected id was found, otherwise 0.
        recall = int(expected_id in search_result_ids)

        results_writer.writerow(
            [
                query_id,
                recall,
                json.dumps(expected_id),
                json.dumps(search_result_ids),
                num_search_result,
                "",
            ]
        )
        results_file.flush()

        all_recalls.append(recall)

    if all_recalls:
        print(f"Overall recall: {sum(all_recalls) / len(all_recalls)}")
    else:  # no expected results
        print("Did not successfully run any test cases, cannot compute overall recall")
    results_file.close()


if __name__ == "__main__":
    asyncio.run(main())
