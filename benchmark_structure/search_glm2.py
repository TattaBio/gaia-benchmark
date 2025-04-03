import os
import faiss
import numpy as np
from torch.nn.functional import normalize
import datasets
import torch
from typing import Literal
from transformers import AutoModel, AutoTokenizer
datasets.disable_caching()

MODEL_NAME = "tattabio/gLM2_650M_embed"
DS_NAME = 'tattabio/scope40_test'


def main(): 
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ds = datasets.load_dataset(DS_NAME)['train']


    def infer_fn(examples):
        sequences = examples["sequence"]
        inputs = tokenizer(
            sequences, return_tensors="pt", padding=True, 
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"].bool()
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(inputs['input_ids'], attention_mask=attention_mask)
            hidden = outputs.pooler_output.float()
            hidden = normalize(hidden, dim=-1)
        examples["cds_features"] = hidden.cpu().numpy()
        return examples


    ds = ds.map(infer_fn, batched=True, batch_size=4, desc="Inference")
    ds.set_format(type="numpy")
    ds = ds.add_faiss_index(column="cds_features", metric_type=faiss.METRIC_INNER_PRODUCT)

    def search_fn(queries, k=100):
        queries = queries.astype(np.float32)
        scores, hits = ds.get_nearest_examples_batch("cds_features", queries, k=k)
        hits_id = [cur_hit['id'] for cur_hit in hits]
        hits_fam = [cur_hit['family'] for cur_hit in hits]
        return {"hits": hits_id, "scores": scores, "hits_family": hits_fam}

    ds = ds.map(search_fn, input_columns=["cds_features"], batch_size=100, batched=True, desc="Searching")
    ds = ds.remove_columns(["cds_features"])

    def reformat_hits(example):
        example = {k:v[0] for k, v in example.items()}
        id = example["id"]
        hits = example["hits"]
        scores = example["scores"]
        output = {'id1': [id] * len(hits),
                        'id2': hits,
                        'score': scores}
        return output

    out_ds = ds.map(reformat_hits, batched=True, batch_size=1, remove_columns=ds.column_names)

    out_name = f"search_result/{os.path.basename(MODEL_NAME)}.tsv"
    out_ds.to_csv(out_name, sep='\t', header=False)


if __name__ == "__main__":
    main()

