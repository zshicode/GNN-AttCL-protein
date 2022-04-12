# This file is used to parse the .fasta file to K-mer indexed corpus.
# Type `python -h` to obtain the concrete usage of this file
# Usage: eg: `python -i inputfile.fasta -c corpus.txt -k 6 -s 2
import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=True, help="data path to seq db (.fasta file)")
parser.add_argument("-k", "--kmer", type=int, default=6, help="k-mer used to index sequence")
parser.add_argument("-s", "--stride", type=int, default=2, help="strides used to index sequence")
parser.add_argument("-c", "--corpus", type=str, required=True, help="the output corpus file")
args = parser.parse_args()

def main():
    handle = open(args.corpus, 'w')
    for seq_record in SeqIO.parse(args.input_file, "fasta"):
        seq_str = str(seq_record.seq).upper()
        for i in range(0, len(seq_str) - args.kmer + 1, args.stride):
            handle.write(seq_str[i:(i+args.kmer)] + ' ')
    handle.close()


if __name__ == "__main__":
    main()
