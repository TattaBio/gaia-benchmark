"""Run gLM2_embed (with contrastive finetune) inference on CDS sequences."""
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
from transformers import AutoModel, default_data_collator
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import get_model_param_count
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs

logger = get_logger(__name__)


def infer(model, eval_dataloader, accelerator, file_path):
    """Runs inference and writes embeddings to the HDF5 file."""
    model.eval()

    # Current index into the h5py dataset.
    curr_idx = 0
    if accelerator.is_main_process:
        f = h5py.File(file_path, 'a')
        emb_dset = f['embeddings']    
    for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            embeds = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'].bool()).pooler_output
            # [batch, layers=1, hidden_size].
            embeds = embeds.unsqueeze(1)
            embeds = accelerator.gather_for_metrics(embeds)
            embeds = embeds.cpu().float().numpy()

        if accelerator.is_main_process:
            # Write embeddings to the h5py dataset.
            num_examples = embeds.shape[0]
            next_idx = curr_idx + num_examples
            emb_dset[curr_idx:next_idx] = embeds
            curr_idx = next_idx
        accelerator.wait_for_everyone()

    return curr_idx


def init_h5py(eval_dataset, file_path: str, dataset_size: int, emb_dim: int, save_ids: bool, id_field: str):
    """Initializes h5py file for storing embeddings."""
    with h5py.File(file_path, 'a') as f:
        # Use num_layers=1 for compatibility with saving multiple layers.
        f.create_dataset('embeddings', shape=(dataset_size, 1, emb_dim), dtype='float32')
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
        default='tattabio/gLM2_650M_embed',
        help="HuggingFace gLM2 model name.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision hash.",
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
    model = AutoModel.from_pretrained(args.model_name, revision=args.revision, trust_remote_code=True)
    accelerator.wait_for_everyone()
    emb_dim = model.config.projection_dim

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
        model, eval_dataloader, accelerator, hdf5_file)

    if accelerator.is_main_process:
        assert num_examples_written == dataset_size

    if accelerator.is_main_process:
        string_of_command = ' '.join(sys.argv)
        # Write info yaml file.
        info_dict = {'command': string_of_command,
                     'dataset_size': dataset_size,
                     'emb_dim': emb_dim}
        info_file = os.path.join(output_dir, "info.yaml")
        with open(info_file, 'w') as f:
            yaml.safe_dump(info_dict, f)
        logger.info(f"HDF5 written to: {hdf5_file}")


if __name__ == "__main__":
    main()