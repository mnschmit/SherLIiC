from typing import Union
import logging
import functools
from tqdm import tqdm
import json
import csv


def init_logging(logger_name):
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger, handler


def log_with_size(*args, logger=None, descr=None):
    ENTRY_MESSAGE = 'Entering {}'
    EXIT_MESSAGE1 = 'Done. Result size is {}.'
    EXIT_MESSAGE2 = 'Done. Result sizes are '

    def _log_with_size(func):
        if logger is None:
            inner_logger = logging.getLogger(func.__module__)
        else:
            inner_logger = logger

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if 'log' in kwargs and not kwargs['log']:
                kwargs.pop('log')
                return func(*args, **kwargs)

            if descr is None:
                inner_logger.info(ENTRY_MESSAGE.format(func.__name__))
            else:
                inner_logger.info(descr)

            f_result = func(*args, **kwargs)

            if isinstance(f_result, tuple):
                exit_message = EXIT_MESSAGE2 + \
                    " ".join(['{}' for _ in range(len(f_result))]) + '.'

                lengths = []
                for res in f_result:
                    try:
                        lengths.append(len(res))
                    except TypeError:
                        lengths.append("")

                inner_logger.info(exit_message.format(*lengths))
            else:
                try:
                    inner_logger.info(EXIT_MESSAGE1.format(len(f_result)))
                except TypeError:
                    inner_logger.warn(
                        "Result of {} has no len()".format(func.__name__)
                    )
                    inner_logger.info("Done")

            return f_result
        return wrapper

    if len(args) == 1 and callable(args[0]):
        return _log_with_size(args[0])
    else:
        return _log_with_size


def get_path(rel_id, relation_index_path="data/relation_index.tsv"):
    global relation_index
    if relation_index is None:
        relation_index = load_relation_index(relation_index_path)

    return relation_index[int(rel_id)]


def voice_of_id(rel_id):
    path = get_path(rel_id)
    if path.startswith("nsubjpass") or path.endswith("nsubjpass") or path.endswith("nsubjpass^-"):
        return "passive"
    else:
        return "active"


def words_from_id(rel_id, relation_index_path="data/relation_index.tsv"):
    rel_path = get_path(rel_id, relation_index_path=relation_index_path)
    return [
        w
        for i, w in enumerate(rel_path.split("___"))
        if i % 2 == 1
    ]


def star_type_to_set(t):
    return frozenset(t.split('*'))


def normalized_star_type(t):
    t_list = list(star_type_to_set(t))
    t_list.sort()
    return '*'.join(t_list)


@log_with_size(descr="Read in rule file.")
def read_in_rules(rule_file):
    rules = {}
    with open(rule_file) as f:
        for line in f:
            premise, conclusion, rel_score,\
                sign_score, ent_support_ratio, *types = line.strip().split("\t")
            if types:
                rules[(premise, conclusion)] = (
                    float(rel_score), float(
                        sign_score), float(ent_support_ratio),
                    types[0], types[1]
                )
            else:
                rules[(premise, conclusion)] = (
                    float(rel_score), float(
                        sign_score), float(ent_support_ratio)
                )
    return rules


@log_with_size(descr="Read in rel ext file.")
def read_in_rel_ext(rel_ext_file, typed_extensions=False):
    rel_ext = {}
    with open(rel_ext_file) as f:
        for line in tqdm(f):
            if typed_extensions:
                sense_number, rel, ext1, ext2, count = line.strip().split("\t")
                rel = combine_type(sense_number, rel)
            else:
                rel, ext1, ext2, count = line.strip().split("\t")

            try:
                rel_ext[rel][(ext1, ext2)] = int(count)
            except KeyError:
                rel_ext[rel] = {(ext1, ext2): int(count)}
    return rel_ext


def combine_type(type_id: Union[str, int], rel_id: Union[str, int]):
    return str(type_id) + "~" + str(rel_id)


@log_with_size(descr="Loading relation index")
def load_relation_index(index_file):
    relation_index = {}
    with open(index_file) as f:
        for line in f:
            idx, rel = line.strip().split("\t")
            relation_index[int(idx)] = rel
    return relation_index


class FilePathDict():
    def __init__(self, json_path):
        self._dict = None
        self._path = json_path
        self._warned_about = set()

    def __getitem__(self, key):
        if self._dict is None:
            with open(self._path) as f:
                self._dict = json.load(f)

        if key in self._dict:
            if self._dict[key] is None or (
                    isinstance(self._dict[key], list)
                    and self._dict[key][0] is None
            ):
                if key not in self._warned_about:
                    logger.warn(
                        (
                            "You have to download resource {}" +
                            " and set the right path to it in file 'file_paths.json'"
                        ).format(key)
                    )
                    self._warned_about.add(key)
                raise KeyError
            else:
                return self._dict[key]
        else:
            logger.warn("Unknown resource name '{}'".format(key))
            raise KeyError


def unpack_rows(rows):
    for row in rows:
        sample_id, prem_type, prem_rel, hypo_type, hypo_rel,\
            fact0, fact1, fact2, fact3,\
            h0, h1, h2, h3,\
            is_prem_reversed, is_hypo_reversed,\
            ex0, ex1, gold_label,\
            rel_score, sign_score, esr_score,\
            num_disagr = row

        premise = (fact0, fact1, fact2, fact3)
        hypothesis = (h0, h1, h2, h3)

        scores = tuple(float(s) for s in (rel_score, sign_score, esr_score))
        yield premise, hypothesis,\
            int(prem_type), int(prem_rel), int(hypo_type), int(hypo_rel),\
            is_prem_reversed == "True", is_hypo_reversed == "True",\
            gold_label == "yes", scores


def read_in_dataset(dataset):
    """
    Output: pr_sent, hy_sent, pr_type, pr_rel, hy_type, hy_rel, pr_rev, hy_rev, goldAns, scores
    """
    with open(dataset) as f:
        cr = csv.reader(f)
        next(cr)  # headers
        for items in unpack_rows(cr):
            yield items


def bl_args(row):
    return row[0:8]


logger, handler = init_logging(__name__)
relation_index = None
