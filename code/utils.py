from typing import List
from commandr import command, Run
from misc import read_in_dataset, combine_type
from tqdm import tqdm

def extract_relations(datasets: List[str], with_types: bool):
    rels = set()
    for d in datasets:
        for row in read_in_dataset(d):
            if with_types:
                pr = combine_type(row[2], row[3])
                hy = combine_type(row[4], row[5])
            else:
                pr = str(row[3])
                hy = str(row[5])
            rels.add(pr)
            rels.add(hy)
    return rels

@command
def filter_embeddings(embedding_file, out_file, dataset=[], types=False):
    """Filters embedding file in word2vec format to keep only embeddings needed in datasets.

    Arguments:
      embedding_file - The embedding file in word2vec format to be filtered.
      out_file  - Filtered version is written to this file.
      dataset - Repeatable argument for all datasets to be considered (default: none).
      types - Whether or not relations in the embedding file are typed (default: false).
    """
    important_relations = extract_relations(dataset, with_types=types)

    num_kept = 0
    with open(embedding_file) as f, open(out_file, 'w') as fout:
        header_line = next(f).strip()
        num_vec, dim_vec = header_line.split()

        print(len(important_relations), dim_vec, file=fout)
        for line in tqdm(f):
            line = line.strip()
            word, *vector = line.split()
            if word in important_relations:
                print(line, file=fout)
                num_kept += 1

    print("Kept {} vectors.".format(num_kept))

if __name__ == "__main__":
    Run()
