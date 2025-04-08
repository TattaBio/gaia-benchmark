import argparse
import datasets
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='tattabio/OG',
        help='Name of the HuggingFace dataset to convert'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='OG_prot90.fasta',
        help='Path to save the output FASTA file'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    sequence_ds = datasets.load_dataset(args.dataset_name)['train']
    seqlist = []

    def to_fasta(ids, seqs):
        records = [SeqRecord(Seq(seq), id=id, description='') for id, seq in zip(ids, seqs)]
        seqlist.extend(records)
    
    sequence_ds.map(to_fasta, input_columns=['id', 'sequence'], batched=True)

    SeqIO.write(seqlist, args.output_file, 'fasta')
    print(f"Wrote {len(seqlist)} sequences to {args.output_file}")


if __name__ == "__main__":
    main()