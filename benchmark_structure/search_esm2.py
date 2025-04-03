import os
import faiss
import numpy as np
from torch.nn.functional import normalize
import datasets
import torch
from typing import Literal
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DS_NAME = 'tattabio/scope40_test'

def main(): 
    model = AutoModel.from_pretrained(MODEL_NAME).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = datasets.load_dataset(DS_NAME)['train']

    def infer_fn(examples):
        sequences = examples["sequence"]
        inputs = tokenizer(sequences, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"].bool()
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(inputs['input_ids'], attention_mask=attention_mask, output_hidden_states=True)
        hiddens = outputs.hidden_states
        layer_idx = len(model.encoder.layer) // 2  # mid layer
        hidden = hiddens[layer_idx].float()
        mask = attention_mask.unsqueeze(-1)
        # Mean pool
        hidden = torch.where(mask, hidden, 0.0)
        hidden = torch.sum(hidden, 1) / torch.sum(mask, dim=1, dtype=hidden.dtype)
        hidden = normalize(hidden, dim=-1)
        examples["cds_features"] = hidden.cpu().numpy()
        return examples

    ds = ds.map(infer_fn, batched=True, batch_size=8, desc="Inference")
    ds = ds.add_faiss_index(column="cds_features", metric_type=faiss.METRIC_INNER_PRODUCT)

    def search_fn(query, k=100):
        query = np.array(query, dtype=np.float32)
        scores, hits = ds.get_nearest_examples("cds_features", query, k=k)
        hits_id = hits['id']
        hits_fam = hits['family']
        return {"hits": hits_id, "scores": scores, "hits_family": hits_fam}

    ds = ds.map(search_fn, input_columns=["cds_features"], desc="Searching")
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

