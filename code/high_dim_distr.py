#!/usr/bin/python3

import numpy as np
from misc import read_in_rel_ext, log_with_size, init_logging
from tqdm import tqdm

logger, handler = init_logging(__name__)

typed_rel_ext = None
typed_feat2idx = None
untyped_rel_ext = None
untyped_feat2idx = None


def weedsPrec(x, y):
    return np.sum(x * (y > 0)) / np.sum(x)


def _cl(x, y):
    return np.minimum(x, y).sum() / np.sum(x)


def invCL(x, y):
    return np.sqrt(_cl(x, y) * (1 - _cl(y, x)))


@log_with_size(descr="Counting features.")
def _count_feats(rel_ext):
    feat2idx = {}
    for rel in tqdm(rel_ext):
        for ent_pair in rel_ext[str(rel)]:
            if ent_pair not in feat2idx:
                feat2idx[ent_pair] = len(feat2idx)

    return feat2idx


def vecForRel(rel, typed=True):
    global typed_rel_ext
    global typed_feat2idx
    global untyped_rel_ext
    global untyped_feat2idx

    rel_ext = typed_rel_ext if typed else untyped_rel_ext    
    if rel_ext is None:
        if typed:
            typed_rel_ext = read_in_rel_ext(
                "data/teg.tsv",
                typed_extensions=True
            )
            typed_feat2idx = _count_feats(typed_rel_ext)
        else:
            untyped_rel_ext = read_in_rel_ext(
                "data/teg-untyped.tsv",
                typed_extensions=False
            )
            untyped_feat2idx = _count_feats(untyped_rel_ext)

    rel_ext, feat2idx = (typed_rel_ext, typed_feat2idx)\
        if typed else (untyped_rel_ext, untyped_feat2idx)

    vec = np.zeros((len(feat2idx),))
    for ent_pair, frq in rel_ext[str(rel)].items():
        vec[feat2idx[ent_pair]] = frq

    return vec
