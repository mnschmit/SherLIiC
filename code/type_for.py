#!/usr/bin/python3

from commandr import command, Run
import json


@command
def id(rel_id, tsg_id=None, type_index="data/type_index.json"):
    with open(type_index) as f:
        tidx = json.load(f)

    if tsg_id is None:
        for tsg in tidx[rel_id].items():
            print(*tsg)
    else:
        print(tidx[rel_id][tsg_id])


@command
def rel(relation, tsg_id=None,
        type_index="data/type_index.json", relation_index="data/relation_index.tsv"):
    rel_idx = {}
    with open(relation_index) as f:
        for line in f:
            rel_id, rel = line.strip().split("\t")
            rel_idx[rel] = rel_id

    id(rel_idx[relation], tsg_id=tsg_id, type_index=type_index)


if __name__ == "__main__":
    Run()
