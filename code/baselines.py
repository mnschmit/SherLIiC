#!/usr/bin/python3

from os import path
import click
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
import csv
import re
import sys
from collections import defaultdict, Counter
import functools
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

from high_dim_distr import weedsPrec, invCL, vecForRel
from misc import voice_of_id, words_from_id, normalized_star_type,\
    init_logging, log_with_size, combine_type, get_path, FilePathDict,\
    read_in_dataset, unpack_rows


logger, handler = init_logging(__name__)

file_paths = FilePathDict("file_paths.json")

ent_graph = None
new_berant = None

patty_pattern_store = None
patty_sub_store = None

ppdb_store = None

schoenmackers_storage = None

chirps_store = None

word2vec_model = None
word2vec_unk_counter = 0
rel_emb_model = None
untyped_rel_emb_model = None

typed_transE_model = None
untyped_transE_model = None
typed_ComplEx_re_model = None
typed_ComplEx_im_model = None
untyped_ComplEx_re_model = None
untyped_ComplEx_im_model = None

missing_resources = set()


@click.group()
def run():
    pass


def lemma(premise, hypothesis,
          type_prem, id_prem, type_hypo, id_hypo,
          is_premise_reversed, is_hypothesis_reversed):
    stop_words = stopwords.words('english')
    pr_lemmata = words_from_id(id_prem)
    hy_lemmata = words_from_id(id_hypo)

    # 1. Criterion: has prem all content words of hypo?
    all_content_words_there = True
    for w in hy_lemmata:
        if w in stop_words:
            continue
        if w not in pr_lemmata:
            all_content_words_there = False
            break

    # 2. Criterion: is predicate the same?
    pr_pred = pr_lemmata[-1] if is_premise_reversed else pr_lemmata[0]
    hy_pred = hy_lemmata[-1] if is_hypothesis_reversed else hy_lemmata[0]
    same_predicate = pr_pred == hy_pred

    # 3. Criterion: is voice and inversement the same?
    voice_pr = voice_of_id(id_prem)
    voice_hy = voice_of_id(id_hypo)
    same_voice = voice_pr == voice_hy
    same_inversement = is_premise_reversed == is_hypothesis_reversed
    third_criterion = same_voice == same_inversement

    return all_content_words_there and same_predicate and third_criterion


def always_yes(*args):
    return True


def majority(*args):
    return False


@log_with_size(descr="Loading Berant I (entailment graph).")
def load_ent_graph():
    global file_paths

    graph = {}
    with open(file_paths["berant"]) as f:
        for line in f:
            prem, edge, hypo = line.strip().split("\t")
            pr_lemmata, pr_type1, pr_type2 = prem[1:-1].split("::")
            hy_lemmata, hy_type1, hy_type2 = hypo[1:-1].split("::")
            inversed = edge == "-R>" or (
                pr_type1 != pr_type2 and pr_type1 == hy_type2
            )
            assert (
                (pr_type1 != hy_type2 or pr_type2 == hy_type1) and
                (pr_type1 == hy_type2 or pr_type2 != hy_type1)
            )

            try:
                graph[(pr_lemmata, hy_lemmata)].append(
                    (pr_type1, pr_type2, inversed)
                )
            except KeyError:
                graph[(pr_lemmata, hy_lemmata)] = [
                    (pr_type1, pr_type2, inversed)
                ]
    return graph


@log_with_size(descr="Loading resource for Berant II.")
def load_new_berant(threshold=0.0):
    global file_paths

    lmtzr = WordNetLemmatizer()

    def list_of_lemmata(string):
        words = string.split()
        if words[0] == "be":
            words[1] = lmtzr.lemmatize(words[1], 'v')
        return words

    db = set()
    with open(file_paths["berant_new"]) as f:
        for line in f:
            prem, hypo, score = line.strip().split("\t")
            if float(score) < threshold:
                break

            prem_inversed = prem.endswith("@R@")
            hypo_inversed = hypo.endswith("@R@")
            if prem_inversed:
                prem = prem[:-3]
            if hypo_inversed:
                hypo = hypo[:-3]
            inversed_rule = prem_inversed != hypo_inversed

            pr_lemmata = " ".join(list_of_lemmata(prem))
            hy_lemmata = " ".join(list_of_lemmata(hypo))

            db.add((pr_lemmata, hy_lemmata, inversed_rule))

    return db


def path2dbformat(rel_id, relation_reversed):
    path = get_path(rel_id)
    words = words_from_id(rel_id)

    if path.startswith("nsubjpass"):
        words.insert(0, "be")

    if path.endswith("nsubjpass") or path.endswith("nsubjpass^-"):
        words.append("be")

    if relation_reversed:
        words.reverse()

    return " ".join(words)


def berant_new(premise, hypothesis,
               type_prem, id_prem, type_hypo, id_hypo,
               is_premise_reversed, is_hypothesis_reversed):
    global new_berant

    if new_berant is None:
        new_berant = load_new_berant()

    prem_lemmata = path2dbformat(id_prem, is_premise_reversed)
    hypo_lemmata = path2dbformat(id_hypo, is_hypothesis_reversed)
    inversed_rule = is_premise_reversed != is_hypothesis_reversed

    return (prem_lemmata, hypo_lemmata, inversed_rule) in new_berant


def berant(premise, hypothesis,
           type_prem, id_prem, type_hypo, id_hypo,
           is_premise_reversed, is_hypothesis_reversed):
    global ent_graph
    if ent_graph is None:
        ent_graph = load_ent_graph()

    prem_lemmata = path2dbformat(id_prem, is_premise_reversed)
    hypo_lemmata = path2dbformat(id_hypo, is_hypothesis_reversed)
    inversed_rule = is_premise_reversed != is_hypothesis_reversed

    if (prem_lemmata, hypo_lemmata) in ent_graph:
        return any([
            inversed_rule == berant_tuple[2]
            for berant_tuple in ent_graph[(prem_lemmata, hypo_lemmata)]
        ])
    else:
        return False


@log_with_size(descr="Loading PATTY ...")
def load_patty():
    global file_paths

    pattern_file, subsumption_file = file_paths["patty"]

    logger.info("Reading PATTY patterns")
    patty_patterns = []
    with open(pattern_file) as f:
        cr = csv.reader(f, delimiter="\t")
        next(cr)  # header
        for row in tqdm(cr):
            pid, ptext, pconf, pdomain, prange = row
            reg_pattern = []
            for pat in ptext.split(";$")[:-1]:
                reg_pattern.append(
                    "(?:" + re.sub(r'\[\[[^]]+\]\]', r'\S+', pat) + ")"
                )
            regex = re.compile("|".join(reg_pattern))
            patty_patterns.append(
                (regex, pdomain.replace('/', '.'), prange.replace('/', '.')))

    logger.info("Reading PATTY subsumptions")
    subsumptions = defaultdict(list)
    with open(subsumption_file) as f:
        cr = csv.reader(f, delimiter="\t")
        next(cr)  # header
        for row in tqdm(cr):
            superpattern, subpattern, conf = row
            subsumptions[int(subpattern)].append(int(superpattern))

    return patty_patterns, subsumptions


def patty(premise, hypothesis,
          type_prem, id_prem, type_hypo, id_hypo,
          is_premise_reversed, is_hypothesis_reversed):
    global patty_pattern_store
    global patty_sub_store
    if patty_pattern_store is None:
        patty_pattern_store, patty_sub_store = load_patty()

    prem_relation = premise[1]
    hypo_relation = hypothesis[1]

    # if arguments are not aligned, Patty cannot help you
    if is_premise_reversed != is_hypothesis_reversed:
        return False

    for pid, (regex, pdomain, prange) in enumerate(patty_pattern_store):
        if regex.search(prem_relation):
            for superpid in patty_sub_store[pid]:
                if patty_pattern_store[superpid][0].search(hypo_relation):
                    return True
    return False


@log_with_size(descr="Loading PPDB-XXXL ...")
def load_ppdb():
    global file_paths

    stop_words = stopwords.words('english')
    interpunction = frozenset(
        [',', '.', '?', '!', ':', "'", '"', ';', '-', '_'])

    storage = set()
    with open(file_paths["ppdb"]) as f:
        cr = csv.reader(f)
        next(cr)  # header
        for row in tqdm(cr):
            prem, hypo = row

            prem_filtered = " ".join([
                w
                for w in prem.split()
                if w not in stop_words and w not in interpunction
            ])
            hypo_filtered = " ".join([
                w
                for w in hypo.split()
                if w not in stop_words and w not in interpunction
            ])

            storage.add((prem_filtered, hypo_filtered))

    return storage


def ppdb(premise, hypothesis,
         type_prem, id_prem, type_hypo, id_hypo,
         is_premise_reversed, is_hypothesis_reversed):
    global ppdb_store
    if ppdb_store is None:
        ppdb_store = load_ppdb()

    stop_words = stopwords.words('english')
    filtered_prem = " ".join([
        w for w in premise[1].split() if w not in stop_words
    ])
    filtered_hypo = " ".join([
        w for w in hypothesis[1].split() if w not in stop_words
    ])

    return filtered_prem == filtered_hypo\
        or (filtered_prem, filtered_hypo) in ppdb_store


@log_with_size(descr="Loading Schoenmackers' rules ...")
def load_schoenmackers():
    global file_paths

    rule_regex = re.compile(
        r'rule\s*"\s*' +
        r'([^(]+)\([^_]+_([AB]), [^_]+_([AB])\)' +
        r'\s*:-\s*([^(]+)\([^_]+_([AB]), [^_]+_([AB])\)' +
        r'[^0-9]*?([0-9]+\.[0-9]+)'
    )
    interpunction = frozenset(
        [',', '.', '?', '!', ':', "'", '"', ';', '-', '_']
    )

    already_there_counter = 0
    storage = set()
    with open(file_paths["schoenmackers"]) as f:
        for line in f:
            m = rule_regex.match(line)
            if m is None:
                print("Error: Found line in wrong format:", file=sys.stderr)
                print(line, file=sys.stderr)
                exit(-1)

            hypo = " ".join([w for w in m.group(1).split()
                             if w not in interpunction])
            h_arg0 = m.group(2)
            h_arg1 = m.group(3)
            prem = " ".join([w for w in m.group(4).split()
                             if w not in interpunction])
            p_arg0 = m.group(5)
            p_arg1 = m.group(6)
            # rel_score = m.group(7)
            assert (h_arg0 == p_arg0) == (h_arg1 == p_arg1)

            triple_to_store = (prem, hypo, h_arg0 == p_arg0)
            if triple_to_store in storage:
                already_there_counter += 1
            storage.add(triple_to_store)

    logger.info("Found {} duplicates".format(already_there_counter))
    return storage


def schoenmackers(premise, hypothesis,
                  type_prem, id_prem, type_hypo, id_hypo,
                  is_premise_reversed, is_hypothesis_reversed):
    global schoenmackers_storage
    if schoenmackers_storage is None:
        schoenmackers_storage = load_schoenmackers()

    pr_lemmata = path2dbformat(id_prem, is_premise_reversed)
    hy_lemmata = path2dbformat(id_hypo, is_hypothesis_reversed)
    alignment_match = is_premise_reversed == is_hypothesis_reversed

    lookup_triple = (pr_lemmata, hy_lemmata, alignment_match)
    return lookup_triple in schoenmackers_storage


@log_with_size(descr="Loading Chirps ...")
def load_chirps():
    global file_paths

    pattern_store = set()
    with open(file_paths["chirps"]) as f:
        for line in f:
            row = line.strip().split("\t")

            # SherLIiC relations only have placeholders at both ends
            if row[0].startswith('{') and row[0].endswith('}')\
               and row[1].startswith('{') and row[1].endswith('}'):
                pat1 = " ".join(row[0].split()[1:-1])
                pat2 = " ".join(row[1].split()[1:-1])

                inversed_rule = not (
                    (row[0].startswith('{a0}') and row[1].startswith('{a0}'))
                    or (row[0].startswith('{a1}') and row[1].startswith('{a1}'))
                )

                pattern_store.add((pat1, pat2, inversed_rule))
    return pattern_store


def chirps(premise, hypothesis,
           type_prem, id_prem, type_hypo, id_hypo,
           is_premise_reversed, is_hypothesis_reversed):
    global chirps_store
    if chirps_store is None:
        chirps_store = load_chirps()

    prem_lemmata = path2dbformat(id_prem, is_premise_reversed)
    hypo_lemmata = path2dbformat(id_hypo, is_hypothesis_reversed)
    inversed_rule = is_premise_reversed != is_hypothesis_reversed

    return (prem_lemmata, hypo_lemmata, inversed_rule) in chirps_store\
        or (hypo_lemmata, prem_lemmata, inversed_rule) in chirps_store


def word2vec_score(premise, hypothesis,
                   type_prem, id_prem, type_hypo, id_hypo,
                   is_premise_reversed, is_hypothesis_reversed):
    global word2vec_model
    global word2vec_unk_counter
    global file_paths
    if word2vec_model is None:
        logger.info("Loading word2vec embeddings.")
        word2vec_model = KeyedVectors.load_word2vec_format(
            file_paths["word2vec"],
            binary=True
        )
        logger.info("Done.")

    def rel2vec(rel_id):
        single_embeddings = []
        for w in words_from_id(rel_id):
            try:
                vec = word2vec_model.get_vector(w)
                single_embeddings.append(vec)
            except KeyError:
                continue
        return sum(single_embeddings) / len(single_embeddings)

    try:
        prem_emb = rel2vec(id_prem)
        hypo_emb = rel2vec(id_hypo)
    except ZeroDivisionError:
        word2vec_unk_counter += 1
        return 0.0

    return (
        prem_emb.dot(hypo_emb)
        / np.linalg.norm(prem_emb)
        / np.linalg.norm(hypo_emb)
    )


def rel_emb_score(premise, hypothesis,
                  type_prem, id_prem, type_hypo, id_hypo,
                  is_premise_reversed, is_hypothesis_reversed):
    global rel_emb_model
    if rel_emb_model is None:
        logger.info("Loading relation embeddings.")
        rel_emb_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/typed_rel_emb.txt",
            binary=False
        )
        logger.info("Done.")
    return rel_emb_model.similarity(
        combine_type(type_prem, id_prem),
        combine_type(type_hypo, id_hypo)
    )


def word2vec_rel_emb_score(*args):
    return word2vec_score(*args) + rel_emb_score(*args)


def untyped_rel_emb_score(premise, hypothesis,
                          type_prem, id_prem, type_hypo, id_hypo,
                          is_premise_reversed, is_hypothesis_reversed):
    global untyped_rel_emb_model
    if untyped_rel_emb_model is None:
        logger.info("Loading untyped relation embeddings.")
        untyped_rel_emb_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/untyped_rel_emb.txt",
            binary=False
        )
        logger.info("Done.")

    return untyped_rel_emb_model.similarity(
        str(id_prem), str(id_hypo),
    )


def word2vec_untyped_rel_emb_score(*args):
    return word2vec_score(*args) + untyped_rel_emb_score(*args)


def transE_typed_score(premise, hypothesis,
                       type_prem, id_prem, type_hypo, id_hypo,
                       is_premise_reversed, is_hypothesis_reversed):
    global typed_transE_model
    if typed_transE_model is None:
        logger.info("Loading typed TransE embeddings.")
        typed_transE_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/typed_transE.txt",
            binary=False
        )
        logger.info("Done.")

    return typed_transE_model.similarity(
        combine_type(type_prem, id_prem),
        combine_type(type_hypo, id_hypo)
    )


def transE_untyped_score(premise, hypothesis,
                         type_prem, id_prem, type_hypo, id_hypo,
                         is_premise_reversed, is_hypothesis_reversed):
    global untyped_transE_model
    if untyped_transE_model is None:
        logger.info("Loading untyped TransE embeddings.")
        untyped_transE_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/untyped_transE.txt",
            binary=False
        )
        logger.info("Done.")

    return untyped_transE_model.similarity(
        str(id_prem), str(id_hypo)
    )


def build_complex_vector(word, re_model, im_model):
    real_vec = re_model.get_vector(str(word))
    im_vec = im_model.get_vector(str(word))
    return real_vec + 1j*im_vec


def complEx_typed_score(premise, hypothesis,
                        type_prem, id_prem, type_hypo, id_hypo,
                        is_premise_reversed, is_hypothesis_reversed):
    global typed_ComplEx_re_model
    global typed_ComplEx_im_model
    if typed_ComplEx_re_model is None:
        logger.info("Loading typed ComplEx embeddings.")
        typed_ComplEx_re_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/typed_ComplEx-real.txt",
            binary=False
        )
        typed_ComplEx_im_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/typed_ComplEx-imaginary.txt",
            binary=False
        )
        logger.info("Done.")

    prem_vec = build_complex_vector(
        combine_type(type_prem, id_prem),
        typed_ComplEx_re_model,
        typed_ComplEx_im_model
    )

    hypo_vec = build_complex_vector(
        combine_type(type_hypo, id_hypo),
        typed_ComplEx_re_model,
        typed_ComplEx_im_model
    )

    return np.real(
        prem_vec.dot(hypo_vec)
        / np.linalg.norm(prem_vec)
        / np.linalg.norm(hypo_vec)
    )


def complEx_untyped_score(premise, hypothesis,
                          type_prem, id_prem, type_hypo, id_hypo,
                          is_premise_reversed, is_hypothesis_reversed):
    global untyped_ComplEx_re_model
    global untyped_ComplEx_im_model
    if untyped_ComplEx_re_model is None:
        logger.info("Loading untyped ComplEx embeddings.")
        untyped_ComplEx_re_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/untyped_ComplEx-real.txt",
            binary=False
        )
        untyped_ComplEx_im_model = KeyedVectors.load_word2vec_format(
            "embeddings/filtered/untyped_ComplEx-imaginary.txt",
            binary=False
        )
        logger.info("Done.")

    prem_vec = build_complex_vector(
        id_prem,
        untyped_ComplEx_re_model,
        untyped_ComplEx_im_model
    )
    hypo_vec = build_complex_vector(
        id_hypo,
        untyped_ComplEx_re_model,
        untyped_ComplEx_im_model
    )

    return np.real(
        prem_vec.dot(hypo_vec)
        / np.linalg.norm(prem_vec)
        / np.linalg.norm(hypo_vec)
    )


tsg_predictions = None
single_type_entries = None


def load_wsiu_predictions(pred_file):
    predictions = {}
    with open(pred_file) as f:
        for line in f:
            tsg, pred = line.strip().split("\t")
            predictions[tsg] = pred == "1"

    single_type_entries = {}
    for tsg, pred in predictions.items():
        if pred:
            delta = 1
        else:
            delta = -1

        for star_t in tsg.split("/"):
            for t in star_t.split('*'):
                single_type_entries[t] = single_type_entries.get(t, 0) + delta

    return predictions, single_type_entries


def word2vec_tsg_score(premise, hypothesis,
                       type_prem, id_prem, type_hypo, id_hypo,
                       is_premise_reversed, is_hypothesis_reversed):
    global tsg_predictions
    global single_type_entries
    global file_paths

    if tsg_predictions is None:
        tsg_predictions, single_type_entries = load_wsiu_predictions(
            file_paths["tsg_preferences"]
        )

    signature = get_normalized_tsg(premise[0][:-3], premise[2][:-3])

    score1 = word2vec_score(
        premise, hypothesis,
        type_prem, id_prem, type_hypo, id_hypo,
        is_premise_reversed, is_hypothesis_reversed
    )

    if signature in tsg_predictions:
        use_typed = tsg_predictions[signature]
    else:
        use_typed_score = sum([
            single_type_entries.get(t, 0)
            for star_t in signature.split("/")
            for t in star_t.split('*')
        ])
        use_typed = use_typed_score > 0

    if use_typed:
        score2 = rel_emb_score(
            premise, hypothesis,
            type_prem, id_prem, type_hypo, id_hypo,
            is_premise_reversed, is_hypothesis_reversed
        )
    else:
        score2 = untyped_rel_emb_score(
            premise, hypothesis,
            type_prem, id_prem, type_hypo, id_hypo,
            is_premise_reversed, is_hypothesis_reversed
        )

    return score1 + score2


def get_vectors(id1, type1, id2, type2, typed=True):
    if typed:
        id1, id2 = combine_type(type1, id1), combine_type(type2, id2)
    return vecForRel(id1, typed=typed), vecForRel(id2, typed=typed)


def typed_weedsPrec_score(premise, hypothesis,
                          type_prem, id_prem, type_hypo, id_hypo,
                          is_premise_reversed, is_hypothesis_reversed):
    return weedsPrec(
        *get_vectors(id_prem, type_prem, id_hypo, type_hypo, typed=True)
    )


def untyped_weedsPrec_score(premise, hypothesis,
                            type_prem, id_prem, type_hypo, id_hypo,
                            is_premise_reversed, is_hypothesis_reversed):
    return weedsPrec(
        *get_vectors(id_prem, type_prem, id_hypo, type_hypo, typed=False)
    )


def typed_invCL_score(premise, hypothesis,
                      type_prem, id_prem, type_hypo, id_hypo,
                      is_premise_reversed, is_hypothesis_reversed):
    return invCL(
        *get_vectors(id_prem, type_prem, id_hypo, type_hypo, typed=True)
    )


def untyped_invCL_score(premise, hypothesis,
                        type_prem, id_prem, type_hypo, id_hypo,
                        is_premise_reversed, is_hypothesis_reversed):
    return invCL(
        *get_vectors(id_prem, type_prem, id_hypo, type_hypo, typed=False)
    )


def all_rules_baseline(*args):
    global bl_name_to_fun
    global missing_resources

    for rbl in ["Lemma", "Berant I", "Berant II",
                "Schoenmackers", "Chirps", "PPDB", "Patty"]:
        if rbl in missing_resources:
            continue

        try:
            pred = bl_name_to_fun[rbl](*args)
        except KeyError:
            logger.warn("Ignoring {} for 'All Rules' baseline.".format(rbl))
            missing_resources.add(rbl)
            continue

        if pred:
            return True
    return False


bl_name_to_fun = {
    "Lemma": lemma,
    "Always yes": always_yes,
    "majority": majority,
    "Berant I": berant,
    "Berant II": berant_new,
    "PPDB": ppdb,
    "Patty": patty,
    "Schoenmackers": schoenmackers,
    "Chirps": chirps,
    "All Rules": all_rules_baseline
}
score_fun = {
    "word2vec": word2vec_score,
    "typed_rel_emb": rel_emb_score,
    "untyped_rel_emb": untyped_rel_emb_score,
    "w2v+typed_rel": word2vec_rel_emb_score,
    "w2v+untyped_rel": word2vec_untyped_rel_emb_score,
    "w2v+tsg_rel_emb": word2vec_tsg_score,
    "WeedsPrec (typed)": typed_weedsPrec_score,
    "WeedsPrec (untyped)": untyped_weedsPrec_score,
    "invCL (typed)": typed_invCL_score,
    "invCL (untyped)": untyped_invCL_score,
    "TransE (untyped)": transE_untyped_score,
    "TransE (typed)": transE_typed_score,
    "ComplEx (untyped)": complEx_untyped_score,
    "ComplEx (typed)": complEx_typed_score,
}


@run.command()
@click.argument(
    'dataset', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--out-prefix', default="non-tunable",
    help="Results will be written to `prefix + dataset file` (default: non-tunable).")
@click.option(
    '-b', '--baseline', "baselines",
    default=[
        "Berant I", "Berant II", "PPDB",
        "Patty", "Schoenmackers", "Chirps",
        "All Rules", "Lemma", "Always yes"
    ],
    multiple=True,
    help="Repeatable argument for each baseline to be evaluated (default: all).",
    type=click.Choice(bl_name_to_fun.keys())
)
@click.option(
    '--use-lemma/--no-lemma', default=True,
    help="Whether to make predictions on top of `Lemma` baseline or not (default: use it).")
def non_tunables(dataset, out_prefix, baselines, use_lemma):
    """Evaluates non-tunable baselines on all DATASETs provided."""

    global logger
    global bl_name_to_fun

    if any([b not in bl_name_to_fun for b in baselines]):
        print("ERROR: Only non-tunable baselines can be specified here.")
        exit(-1)

    logger.info("On top of lemma: {}".format(use_lemma))

    for d in dataset:
        output = []
        logger.info(
            "Evaluating {} non-tunable baselines on {}".format(len(baselines), d))
        for bl in baselines:
            truth, pred = run_baseline(
                d, method=bl, use_lemma=use_lemma
            )
            output.append(
                (
                    bl,
                    precision_score(truth, pred),
                    recall_score(truth, pred),
                    f1_score(truth, pred)
                )
            )
            logger.info("Finished {} baseline.".format(bl))

        out_file_path = "{}-{}.txt".format(
            out_prefix,
            path.splitext(path.basename(d))[0]
        )
        with open(out_file_path, 'w') as f:
            for out_tupel in output:
                print(
                    "{}\t{:.3f}:{:.3f}:{:.3f}".format(*out_tupel),
                    file=f
                )


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fitted_ = False

    def check_input(self, X):
        if any([len(x) != 1 for x in X]):
            raise ValueError("Every sample should have exactly one feature.")

    def fit(self, X, y):
        self.check_input(X)

        scores = [x[0] for x in X]
        pre, rec, thr = precision_recall_curve(y, scores)
        f1 = [
            2 * p * r / (p+r) if (p+r) > 0.0 else 0.0
            for p, r in zip(pre, rec)
        ]

        self.thr_ = max(zip(f1, thr), key=lambda x: x[0])[1]
        self.fitted_ = True

        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("You have to call fit() first.")
        self.check_input(X)

        scores = [x[0] for x in X]
        pred = [s >= self.thr_ for s in scores]
        return pred


def obtain_score_truth_from_dataset(
        dataset, method,
        use_lemma=True, qualitative=False
):
    global word2vec_unk_counter
    if method == "word2vec":
        word2vec_unk_counter = 0

    lemma_predictions = []
    scores = []
    truth = []
    if qualitative:
        samples = []
    for pr, hy, type_pr, id_pr, type_hy, id_hy,\
        reversed_pr, reversed_hy, is_entailment,\
            sh_scores in read_in_dataset(dataset):
        method_args = [
            pr, hy,
            type_pr, id_pr, type_hy, id_hy,
            reversed_pr, reversed_hy
        ]

        if use_lemma:
            lemma_predictions.append(lemma(*method_args))

        if method == "Sherlock+ESR":
            score = functools.reduce(lambda a, b: a*b, sh_scores)
        else:
            score = score_fun[method](*method_args)

        scores.append(score)
        truth.append(is_entailment)

        if qualitative:
            path_pr = get_path(id_pr)
            path_hy = get_path(id_hy)
            samples.append((path_pr, path_hy))

    if use_lemma:
        lemma_score = max(scores)
        scores = [
            lemma_score if lemma_pred else s
            for s, lemma_pred in zip(scores, lemma_predictions)
        ]

    if method == "word2vec":
        logger.info("word2vec UNKs: {} (on {})".format(
            word2vec_unk_counter, dataset
        ))

    if qualitative:
        return scores, truth, samples
    else:
        return scores, truth


@run.command()
@click.argument("dev_dataset")
@click.argument("out_file")
@click.argument("methods", nargs=-1)
def error_analysis(dev_dataset, out_file, methods):
    """Identifies errors made by all of the specified tunable methods and writes the result to OUT_FILE."""

    global score_fun
    global logger

    logger.info("Analyze results from {} baselines.".format(len(methods)))
    if any([m not in score_fun for m in methods]):
        print("ERROR: Only tunable baselines are allowed for error analysis.")
        exit(-1)

    wrongs_per_method = defaultdict(set)
    for method in methods:
        dev_scores, dev_truth, samples = obtain_score_truth_from_dataset(
            dev_dataset, method, use_lemma=True, qualitative=True
        )

        clf = ThresholdClassifier()
        dev_feats = np.array(dev_scores).reshape(-1, 1)
        clf.fit(dev_feats, dev_truth)
        dev_pred = clf.predict(dev_feats)

        for p, t, rule in zip(dev_pred, dev_truth, samples):
            if p != t:
                wrongs_per_method[method].add((rule, t))

    if wrongs_per_method:
        all_wrong = functools.reduce(
            lambda a, b: a & b,
            wrongs_per_method.values()
        )
    else:
        all_wrong = set()

    with open(out_file, 'w') as fout:
        print("Prem", "Hypo", "Truth", sep="\t", file=fout)
        for rule, t in all_wrong:
            print(rule[0], rule[1], t, sep="\t", file=fout)


def evaluate_dev_test_curve(dev_dataset, test_dataset, method, use_lemma=True):
    global logger

    logger.info("Start computing scores for method {}".format(method))
    dev_scores, dev_truth = obtain_score_truth_from_dataset(
        dev_dataset, method, use_lemma=use_lemma
    )
    test_scores, test_truth = obtain_score_truth_from_dataset(
        test_dataset, method, use_lemma=use_lemma
    )
    logger.info("Done computing scores for method {}".format(method))

    clf = ThresholdClassifier()
    dev_feats = np.array(dev_scores).reshape(-1, 1)
    clf.fit(dev_feats, dev_truth)

    dev_pred = clf.predict(dev_feats)
    dev_prec = precision_score(dev_truth, dev_pred)
    dev_rec = recall_score(dev_truth, dev_pred)
    dev_f1 = f1_score(dev_truth, dev_pred)

    test_feats = np.array(test_scores).reshape(-1, 1)
    test_pred = clf.predict(test_feats)
    test_prec = precision_score(test_truth, test_pred)
    test_rec = recall_score(test_truth, test_pred)
    test_f1 = f1_score(test_truth, test_pred)

    return clf.thr_, (dev_prec, dev_rec, dev_f1), (test_prec, test_rec, test_f1)


@run.command()
@click.argument("dev")
@click.argument("test")
@click.option(
    "-o", "--out-file", default="tunable-devtest.txt",
    help="Results will be written to this file (default: tunable-devtest.txt)."
)
@click.option(
    "-b", "--baselines", multiple=True,
    default=[
        "word2vec", "typed_rel_emb", "untyped_rel_emb",
        "w2v+typed_rel", "w2v+untyped_rel",
        "w2v+tsg_rel_emb",
        "WeedsPrec (typed)", "WeedsPrec (untyped)",
        "invCL (typed)", "invCL (untyped)",
        "TransE (typed)", "TransE (untyped)",
        "ComplEx (typed)", "ComplEx (untyped)",
        "Sherlock+ESR"
    ]
)
@click.option(
    '--use-lemma/--no-lemma', default=True,
    help="Whether to make predictions on top of `Lemma` baseline or not (default: use it).")
def tunables(dev, test, out_file, baselines, use_lemma):
    """Evaluates tunable baselines by tuning them on DEV and measuring performance on DEV and TEST."""
    global logger

    logger.info("On top of lemma: {}".format(use_lemma))
    logger.info("Start evaluating {} tunable baselines".format(len(baselines)))

    with open(out_file, 'w') as fout:
        for method in baselines:
            thr, dev_metrics, test_metrics = evaluate_dev_test_curve(
                dev, test, method,
                use_lemma=use_lemma
            )
            print(
                method,
                (":".join(["{:.3f}"]*7)).format(
                    thr,
                    dev_metrics[0], dev_metrics[1], dev_metrics[2],
                    test_metrics[0], test_metrics[1], test_metrics[2]
                ),
                sep="\t", file=fout
            )


def print_metrics(truth, pred, only_f1=False, round_it=True):
    prec = precision_score(truth, pred)
    recall = recall_score(truth, pred)
    f1 = f1_score(truth, pred)
    if only_f1:
        names = ["F1-Score"]
        numbers = [f1]
    else:
        names = ["Precision", "Recall", "F1-Score"]
        numbers = [prec, recall, f1]

    for name, num in zip(names, numbers):
        if round_it:
            print("{}: {:.3f}".format(name, num))
        else:
            print("{}: {}".format(name, num))


def print_examples(predictions, k=5):
    sample = random.sample(predictions, k) if len(
        predictions) > k else predictions
    for rule in sample:
        print(*rule, sep=" => ")


def tunable_predictor(method, threshold):
    def predict(*args):
        return score_fun[method](*args) >= threshold
    return predict


def run_baseline(dataset, method, examples=False, use_lemma=True, threshold=0.5):
    global bl_name_to_fun
    global word2vec_unk_counter
    if method == "word2vec":
        word2vec_unk_counter = 0

    predictor = None
    if method in bl_name_to_fun:
        predictor = bl_name_to_fun[method]
    elif method in score_fun:
        predictor = tunable_predictor(method, threshold)
    else:
        raise ValueError("Unknown method name: {}".format(method))

    pred = []
    truth = []
    if examples:
        correct_positives = []
        correct_negatives = []
        false_positives = []
        false_negatives = []

    logger.info("Evaluating {} on {}.".format(method, dataset))
    if method in ["PPDB", "Patty", "All Rules"]:
        logger.info("This might take a while.")

    for pr, hy,\
            type_pr, id_pr, type_hy, id_hy,\
            reversed_pr, reversed_hy,\
            is_entailment, sh_scores in read_in_dataset(dataset):
        method_args = [
            pr, hy, type_pr, id_pr, type_hy, id_hy,
            reversed_pr, reversed_hy
        ]

        if use_lemma and lemma(*method_args):
            pred.append(True)
        else:
            try:
                pred.append(predictor(*method_args))
            except KeyError:
                logger.error("Cannot produce results for {}".format(method))
                if examples:
                    return [], [], []
                else:
                    return [], []

        truth.append(is_entailment)

        if examples:
            path_pr = get_path(id_pr)
            path_hy = get_path(id_hy)
            if pred[-1] == truth[-1]:
                if pred[-1]:
                    correct_positives.append((path_pr, path_hy))
                else:
                    correct_negatives.append((path_pr, path_hy))
            else:
                if pred[-1]:
                    false_positives.append((path_pr, path_hy))
                else:
                    false_negatives.append((path_pr, path_hy))

    if method == "word2vec":
        logger.info("word2vec UNKs: {}".format(word2vec_unk_counter))

    if examples:
        return truth, pred,\
            [correct_positives, correct_negatives,
                false_positives, false_negatives]
    else:
        return truth, pred


@run.command()
@click.argument('baseline', type=click.Choice(bl_name_to_fun.keys() | score_fun.keys()))
@click.argument('dataset', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--examples/--no-examples", default=False,
    help="Whether or not to print example predictions (default: false)."
)
@click.option(
    '--use-lemma/--no-lemma', default=True,
    help="Whether to make predictions on top of `Lemma` baseline or not (default: use it).")
@click.option(
    '--rounding/--no-rounding', default=True,
    help="Whether or not results should be rounded (default: true)."
)
@click.option(
    "-t", "--threshold", default=0.5,
    help="Threshold for tunable baselines (default: 0.5); ignored for non-tunable baselines."
)
def single(baseline, dataset, examples, use_lemma, rounding, threshold):
    """Runs BASELINE on (a list of) DATASETs
    (See `list_baselines` for a list of available baselines).
    """

    for d in dataset:
        results = run_baseline(
            d, baseline, examples=examples,
            use_lemma=use_lemma, threshold=threshold
        )

        truth, pred = results[0:2]

        print("{} baseline on {}:".format(baseline, d))
        print_metrics(truth, pred, round_it=rounding)

        if examples:
            cp, cn, fp, fn = results[2]
            print("True positives:")
            print_examples(cp)
            print("True negatives:")
            print_examples(cn)
            print("False positives:")
            print_examples(fp)
            print("False negatives:")
            print_examples(fn)


@run.command()
@click.argument("event_graph_file")
@click.argument("train_file")
@click.option(
    "--types/--no-types", default=True,
    help="Whether or not the input event graph is typed (default: typed)."
)
@click.option(
    "--entity-pairs/--single-entities", default=False,
    help="Whether to use entity pair features for embedding learning" +
    " or treat entities as two distinct features of a relation (default: the latter)."
)
def create_training(event_graph_file, train_file, types, entity_pairs):
    """Converts an event graph to a training corpus for word2vec."""
    global logger

    logger.info("Creating embedding training corpus")
    with open(event_graph_file) as f, open(train_file, 'w') as fout:
        for line in tqdm(f):
            line = line.strip().split("\t")
            if not types:
                rel_id, arg0, arg1 = line[0], line[1], line[2]
                frq = int(line[3])
            else:
                rel_id = line[0] + "~" + line[1]
                arg0, arg1 = line[2], line[3]
                frq = int(line[4])

            for _ in range(frq):
                if entity_pairs:
                    print(
                        rel_id,
                        arg0 + "_ENTSEP_" + arg1,
                        file=fout
                    )
                else:
                    print(
                        "arg0:"+arg0,
                        rel_id,
                        "arg1:"+arg1,
                        file=fout
                    )


@run.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("baseline", type=click.Choice(score_fun.keys()))
@click.option(
    '--use-lemma/--no-lemma', default=True,
    help="Whether to make predictions on top of `Lemma` baseline or not (default: use it).")
def find_threshold(dataset, baseline, use_lemma):
    """Computes F1-optimal threshold for given tunable baseline and dataset."""
    global logger

    logger.info("Computing best threshold for {}".format(baseline))
    scores, truth = obtain_score_truth_from_dataset(
        dataset, baseline, use_lemma=use_lemma
    )
    feats = np.array(scores).reshape(-1, 1)

    logger.info("Start optimizing F1")
    clf = ThresholdClassifier()
    clf.fit(feats, truth)
    print(clf.thr_)


@run.command("tsg_pref")
@click.argument("dataset", click.Path(exists=True, dir_okay=False))
@click.argument("out_file", click.File('w'))
@click.option(
    "-t", "--threshold", default=0.0,
    help="How much do typed embeddings have to be better than" +
    " untyped embeddings to be preferred? (default: 0.0)"
)
def tsg_pref(dataset, out_file, thr=0.0):
    """Determines which type signatures benefit from typed embeddings.

    Arguments:
      dataset - The dataset to determine type signature preferences.
      out_file - Results will be written to this file.
    """

    rowsPerSignature = sort_rows_by_signature(dataset)

    for sign in tqdm(rowsPerSignature):
        truth = generate_truth_from_rows(rowsPerSignature[sign])
        pred = estimate_prediction_from_truth(truth, thr)
        print(sign, pred, sep="\t", file=out_file)

    logger.info("I saw {} type signatures.".format(len(rowsPerSignature)))


def sort_rows_by_signature(dataset):
    rowsPerSignature = {}
    with open(dataset) as f:
        cr = csv.reader(f)
        next(cr)  # headers
        for row in cr:
            signature = get_normalized_tsg(row[5][:-3], row[7][:-3])

            try:
                rowsPerSignature[signature].append(row)
            except KeyError:
                rowsPerSignature[signature] = [row]

    return rowsPerSignature


def get_normalized_tsg(t1, t2):
    type0 = normalized_star_type(t1)
    type1 = normalized_star_type(t2)
    return "/".join(sorted((type0, type1)))


def generate_truth_from_rows(rows):
    truth_for_rows = []
    for pr, hy, type_pr, id_pr, type_hy, id_hy,\
            reversed_pr, reversed_hy,\
            is_entailment, sh_scores in unpack_rows(rows):
        method_args = [
            pr, hy, type_pr, id_pr, type_hy, id_hy,
            reversed_pr, reversed_hy
        ]

        score0 = score_fun["untyped_rel_emb"](*method_args)
        score1 = score_fun["typed_rel_emb"](*method_args)

        if is_entailment:
            truth_for_rows.append(0 if score0 > score1 else 1)
        else:
            truth_for_rows.append(1 if score0 > score1 else 0)
    return truth_for_rows


def estimate_prediction_from_truth(truth, thr):
    c = Counter(truth)
    ratio0 = c[0] / sum(c.values())
    ratio1 = c[1] / sum(c.values())
    if ratio1 - ratio0 > thr:
        return 1
    else:
        return 0


@run.command()
def list_baselines():
    """Lists all available baselines.
    """

    global bl_name_to_fun
    global score_fun

    print("Non-tunable baselines:")
    for bl in bl_name_to_fun:
        print("*", bl)
    print()

    print("Tunable baselines:")
    for bl in score_fun:
        print("*", bl)


if __name__ == "__main__":
    run()
