"""Run ESM inference on CDS sequences."""
from typing import Literal
import argparse
import logging
import os
from os.path import basename
import sys
import yaml
import h5py
import torch
from datetime import timedelta
from tqdm.auto import tqdm
import datasets
from datasets import load_from_disk, concatenate_datasets
from transformers import default_data_collator, AutoModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import get_model_param_count
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs

logger = get_logger(__name__)


class PoolLayer(torch.nn.Module):
    """Applies pooling on protein sequence embeddings."""

    def __init__(self, method: Literal["mean", "max"] = "mean") -> None:
        """Initialize pooling method."""
        self.method = method
        if self.method not in ["mean", "max"]:
            raise ValueError("Method must be either 'mean' or 'max'.")
        super().__init__()

    def forward(self, embeds: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        """Applies pooling.

        Args:
            embeds: [..., seq_len, hidden_dim].
            mask: [..., seq_len].

        Returns:
            Outputs of shape [..., hidden_dim].
        """
        # [..., seq_len, 1].
        mask = mask.unsqueeze(-1)
        if self.method == "mean":
            embeds = torch.where(mask, embeds, 0.0)
            # Mean pool across the seq len axis.
            embeds = torch.sum(embeds, -2)
            embeds /= torch.clamp(torch.sum(mask, dim=-2,
                                  dtype=embeds.dtype), min=1.0)
        elif self.method == "max":
            # Set invalid to large negative val.
            embeds = torch.where(mask, embeds, torch.finfo(embeds.dtype).min)
            embeds = torch.max(embeds, dim=-2).values
            # Set zero embeddings where there are no valid elements in the sequence.
            embeds = torch.where(mask.any(-2),
                                 embeds, torch.zeros_like(embeds))
        return embeds


def infer(model, eval_dataloader, accelerator, layers_to_save, file_path):
    """Runs inference and writes embeddings to the HDF5 file."""
    pooler = PoolLayer(method="mean")
    embedding_dict = {}

    def get_embedding_hook(layer: int):
        # Hook to store the embeddings.
        def hook(model, input, output):
            # Unpack inputs according to modeling_esm.EsmLayer.forward
            x_in, attention_mask = input[:2]
            # EsmLayer returns a tuple. Get only the first output.
            output = output[0]
            # Apply pooling across the CDS sequence.
            output = pooler(output, mask=attention_mask)
            embedding_dict[layer] = output
        return hook

    # Apply the hooks.
    unwrapped_model = accelerator.unwrap_model(model)
    for idx in layers_to_save:
        module = unwrapped_model.encoder.layer[idx]
        module.register_forward_hook(get_embedding_hook(idx))

    model.eval()
    # Current index into the h5py dataset.
    curr_idx = 0
    if accelerator.is_main_process:
        f = h5py.File(file_path, 'a')
        emb_dset = f['embeddings']    
    for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            _ = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'].bool())
            embeds = [embedding_dict[l] for l in layers_to_save]
            # [batch, hidden_layers, hidden_size].
            embeds = torch.stack(embeds, dim=-2)
            embeds = accelerator.gather_for_metrics(embeds)
            embeds = embeds.cpu().float().detach().numpy()
            embedding_dict.clear()

        if accelerator.is_main_process:
            # Write embeddings to the h5py dataset.
            num_examples = embeds.shape[0]
            next_idx = curr_idx + num_examples
            emb_dset[curr_idx:next_idx] = embeds
            curr_idx = next_idx
        accelerator.wait_for_everyone()

    return curr_idx


def init_h5py(eval_dataset, file_path: str, dataset_size: int, emb_dim: int, num_layers: int, save_ids: bool, id_field: str):
    """Initializes h5py file for storing embeddings."""
    with h5py.File(file_path, 'a') as f:
        f.create_dataset('embeddings', shape=(
            dataset_size, num_layers, emb_dim), dtype='float32')
        if save_ids:
            f.create_dataset('ids', shape=(dataset_size,),
                             dtype=h5py.special_dtype(vlen=str))
            logger.info("***** Populating ids *****")
            ids_array = f['ids']
            ids_array[...] = eval_dataset[id_field]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The path to the tokenized CDS dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='facebook/esm2_t33_650M_UR50D',
        help="HuggingFace ESM model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory to store inference outputs and embeddings."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per device batch size for inference.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default='last',
        help="Layer to save. Comma separated list of integers or 'mid' and 'last'. Default is 'last'"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--save_ids",
        type=bool,
        default=False,
        help="If true, save the protein ids to the h5py file.",
    )
    parser.add_argument(
        "--id_field",
        type=str,
        default='id',
        help="Dataset CDS id field name.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Define the output dir.
    dir_name = f"{basename(args.model_name)}_{basename(args.dataset_path)}_embeds"
    output_dir = os.path.join(args.output_dir, dir_name)

    # Initialize Accelerator.
    project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=output_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200000))
    accelerator = Accelerator(
        project_config=project_config, kwargs_handlers=[kwargs])

    # Create the output directory.
    if accelerator.is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("***** Load Dataset *****")
    eval_dataset = load_from_disk(args.dataset_path)
    logger.info("***** Finished Loading Dataset *****")
    # If the dataset contains multiple splits, concatenate them.
    if type(eval_dataset) is datasets.DatasetDict:
        splits = list(eval_dataset.keys())
        if len(splits) > 1:
            logger.info(f"Concatenating splits: {splits}")
            eval_dataset = concatenate_datasets(
                list(eval_dataset.values()))
        else:
            eval_dataset = eval_dataset[splits[0]]

    # Initialize model.
    model = AutoModel.from_pretrained(args.model_name)
    num_layers = model.config.num_hidden_layers
    emb_dim = model.config.hidden_size

    # Parse the layer to save.
    mid_layer = (num_layers // 2) - 1
    last_layer = num_layers - 1
    if args.layers == "mid":
        layers = [mid_layer]
    elif args.layers == "last":
        layers = [last_layer]
    else:
        try:
            layers = [int(layer) for layer in layers.split(",")]
        except ValueError:
            raise ValueError("Layers must be a list of integers.")
    num_layers_to_save = len(layers)
    max_layer_idx = max(layers)

    # To speed up inference, remove the unused layers from the model.
    model.encoder.layer = model.encoder.layer[:max_layer_idx + 1]

    accelerator.wait_for_everyone()

    # Initialize dataLoader.
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Initialize HDF5.
    dataset_size = len(eval_dataset)
    logger.info(f"Embedding dataset size: {dataset_size}")
    hdf5_file = os.path.join(output_dir, "cds_embeddings.h5")
    logger.info(f"Creating output h5py file {hdf5_file}")
    if accelerator.is_main_process:
        init_h5py(
            eval_dataset=eval_dataset,
            file_path=hdf5_file,
            dataset_size=dataset_size,
            emb_dim=emb_dim,
            num_layers=num_layers_to_save,
            save_ids=args.save_ids,
            id_field=args.id_field,
        )
        logger.info("H5py file created")

    accelerator.wait_for_everyone()

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    logger.info("***** Inferring samples *****")
    logger.info(f"Num examples = {len(eval_dataset)}")
    logger.info(f"Instantaneous batch size per device = {args.batch_size}")
    accelerator.print(f"Loaded model from : {args.model_name}")
    logger.info(f"Number of parameters = {get_model_param_count(model):,}")

    # Run inference!
    num_examples_written = infer(
        model, eval_dataloader, accelerator, layers, hdf5_file)

    if accelerator.is_main_process:
        assert num_examples_written == dataset_size

    if accelerator.is_main_process:
        string_of_command = ' '.join(sys.argv)
        # Write info yaml file.
        info_dict = {'command': string_of_command,
                     'dataset_size': dataset_size,
                     'emb_dim': emb_dim,
                     'num_layers': num_layers,
                     'layers_to_save': layers}
        info_file = os.path.join(output_dir, "info.yaml")
        with open(info_file, 'w') as f:
            yaml.safe_dump(info_dict, f)
        logger.info(f"HDF5 written to: {hdf5_file}")


if __name__ == "__main__":
    main()