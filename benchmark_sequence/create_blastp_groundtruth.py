import argparse                                                                                                                                                                              
import os                                                                                                                                                                                    
import subprocess                                                                                                                                                                            
import sys                                                                                                                                                                                   
from datetime import datetime                                                                                                                                                                
import pandas as pd

def print_progress(message):                                                                                                                                                                 
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")                                                                                                                     
                                                                                                                                                                                          
def run_command(command):                                                                                                                                                                    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)                                                                                          
    stdout, stderr = process.communicate()                                                                                                                                                   
    if process.returncode != 0:                                                                                                                                                              
        print(f"Error executing command: {command}")                                                                                                                                         
        print(f"stderr: {stderr.decode()}")                                                                                                                                                  
        sys.exit(1)                                                                                                                                                                          
    return stdout.decode()                                                                                                                                                                   
                                                                                                                                                                                          
def run_blastp(db_path, query_path, output_path, threads):
    blast_command = (
        f'blastp -outfmt "6 sseqid qseqid pident qlen slen length mismatch gapopen '
        f'qstart qend sstart send sseq evalue bitscore" -db {db_path} -num_threads {threads} '
        f'-query {query_path} -out {output_path}'
    )
    run_command(blast_command)
    
    # Add headers to the BLAST output
    header = ['sseqid', 'qseqid', 'pident', 'qlen', 'slen', 'length', 'mismatch', 
              'gapopen', 'qstart', 'qend', 'sstart', 'send', 'sseq', 'evalue', 'bitscore']
    hits = pd.read_csv(output_path, sep='\t', names=header)
    hits.to_csv(output_path, index=False, sep='\t')

def reformat_blastp_tsv(input_file, output_file, threshold=75, qcov_threshold=0.7, scov_threshold=0.7, top_n=100):
    matches_by_query = {}

    with open(input_file, 'r') as infile:
        # Skip the header line in the input file
        next(infile)

        for line in infile:
            parts = line.strip().split("\t")
            target, query, pident, qlen, slen, length, mismatch, gapopen, qstart, qend, sstart, send, sseq, evalue, bits = parts

            pident = float(pident)
            bits = float(bits)
            qcov = (int(qend) - int(qstart) + 1) / float(qlen)
            scov = (int(send) - int(sstart) + 1) / float(slen)

            # Exclude self-hits
            if query == target:
                continue

            # Apply sequence identity, qcov, and scov thresholds
            if pident >= threshold and qcov >= qcov_threshold and scov >= scov_threshold:
                evalue_bit_score = f"{evalue}/{bits}"

                if query not in matches_by_query:
                    matches_by_query[query] = []

                matches_by_query[query].append((target, bits, pident, qcov, scov, evalue_bit_score))

    # Write output
    with open(output_file, 'w') as outfile:
        # Write header to output file
        outfile.write("query_sequence_id\tmatched_sequence_ids\talignment_identities\tqcov\tscov\tevalue_bit_score\n")

        for query, matches in matches_by_query.items():
            # Sort matches by bitscore in descending order and select top_n
            matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)[:top_n]
            
            target_ids = ",".join([m[0] for m in matches_sorted])
            pidents = ",".join([f"{m[2]:.2f}" for m in matches_sorted])
            qcovs = ",".join([f"{m[3]:.2f}" for m in matches_sorted])
            scovs = ",".join([f"{m[4]:.2f}" for m in matches_sorted])
            evalue_bit_scores = ",".join([m[5] for m in matches_sorted])
            
            outfile.write(f"{query}\t{target_ids}\t{pidents}\t{qcovs}\t{scovs}\t{evalue_bit_scores}\n")

def main():                                                                                                                                                                                  
    parser = argparse.ArgumentParser(description="Process FASTA files for sequence analysis.")                                                                                               
    parser.add_argument("--query_fasta", required=True, help="Path to the query FASTA file")
    parser.add_argument("--target_fasta", required=True, help="Path to the target FASTA file")
    parser.add_argument("--output_dir", required=True, help="Path to the working directory")                                                                                                   
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to use (default: 30)")                                                                                                                                                              
    parser.add_argument("--threshold", type=float, default=75, help="Sequence identity threshold (default: 75)")
    parser.add_argument("--qcov_threshold", type=float, default=0.7, help="Query coverage threshold (default: 0.7)")
    parser.add_argument("--scov_threshold", type=float, default=0.7, help="Subject coverage threshold (default: 0.7)")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top matches to keep (default: 100)")
    args = parser.parse_args()                                                                                                                                                               

    os.makedirs(args.output_dir, exist_ok=True)
    # Set paths                                                                                                                                                                         
    BLAST_OUTPUT = os.path.join(args.output_dir, "blastp_raw_output.tsv")                                                                                                                          
    PROCESSED_OUTPUT = os.path.join(args.output_dir, "blastp_groundtruth_results.tsv")
    BLAST_DB = os.path.join(args.output_dir, "blastdb")                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    # Check if BLAST database exists, create if not                                                                                                                                          
    if not os.path.exists(f"{BLAST_DB}.pin"):                                                                                                                                                
        print_progress("BLAST database not found. Creating...")                                                                                                                              
        run_command(f"makeblastdb -in {args.target_fasta} -dbtype prot -out {BLAST_DB}")                                                                                                           
        print_progress("BLAST database created.")                                                                                                                                            
    else:                                                                                                                                                                                    
        print_progress("BLAST database found.")                                                                                                                                              
                                                                                                                                                                                            
    # Run BLASTp                                                                                                                                                                             
    print_progress("Running BLASTp...")                                                                                                                                                      
    run_blastp(BLAST_DB, args.query_fasta, BLAST_OUTPUT, str(args.threads))
    print_progress("BLASTp completed.")                                                                                                                                                      
                                                                                                                                                                                            
    # Reformat the BLAST results
    print_progress("Reformatting BLAST results...")
    reformat_blastp_tsv(
        BLAST_OUTPUT, 
        PROCESSED_OUTPUT,
        threshold=args.threshold,
        qcov_threshold=args.qcov_threshold,
        scov_threshold=args.scov_threshold,
        top_n=args.top_n
    )
    print_progress("Reformatting completed.")

    print_progress("Pipeline completed successfully!")                                                                                                                                       
                                                                                                                                                                                            
if __name__ == "__main__":                                                                                                                                                                   
 main() 