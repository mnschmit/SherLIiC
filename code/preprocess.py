#!/usr/bin/python3

from commandr import command, Run
from tqdm import tqdm
import gzip
import csv


@command
def ppdb(in_file, out_file="ppdb.csv"):
    """Takes ppdb-2.0-xxxl-all.gz and converts it to csv.

    Arguments:
      in_file - Path to ppdb-2.0-xxxl-all.gz
      out_file - Output file (default: ppdb.csv)
    """

    with gzip.open(in_file, 'rt') as f, open(out_file, 'w') as fout:
        cw = csv.writer(fout)
        cw.writerow(["source", "target"])

        for line in tqdm(f):
            entry = line.strip().split(" ||| ")
            source = entry[1]
            target = entry[2]
            ent = entry[5]
            if ent == "Equivalence":
                cw.writerow([source, target])
                cw.writerow([target, source])
            elif ent == "ForwardEntailment":
                cw.writerow([source, target])
            elif ent == "ReverseEntailment":
                cw.writerow([target, source])


if __name__ == "__main__":
    Run()
