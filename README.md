# SherLIiC
## Typed Event Graph and Lexical Inference Benchmark

This is the data and code for the paper:

<b>"SherLIiC: A Typed Event-Focused Lexical Inference Benchmark for Evaluating Natural Language Inference"</b><br/>
Martin Schmitt and Hinrich Schütze. ACL 2019. [paper](https://www.aclweb.org/anthology/P19-1086)

Additional material (e.g., slides of the talk) can be found [here](https://www.cis.uni-muenchen.de/~martin/publ_files/sherliic_acl2019.html).

```
@inproceedings{schmitt2019sherliic,
    title = "{S}her{LI}i{C}: A Typed Event-Focused Lexical Inference Benchmark for Evaluating Natural Language Inference",
    author = {Schmitt, Martin  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1086",
    pages = "902--914"
}
```

***

# How to get the data
## SherLIiC resources
The SherLIiC resources can be downloaded from [here](http://cistern.cis.lmu.de/SherLIiC/sherliic.tar.gz).

**You should extract the archive to the folder `data`.**

## Full embedding files
The embedding files in `embeddings/filtered` only contain embeddings for the relations in SherLIiC-dev and SherLIiC-test.

Full embedding files (i.e. embeddings for all entities and relations in the whole SherLIiC event graph) can be downloaded [here](http://cistern.cis.lmu.de/SherLIiC/embeddings.tar.gz).

# How to use the code

## List all available baselines
To see a list of all available baselines, run:
```
python3 code/baselines.py list_baselines
```

## Run a single baseline
```
python3 code/baselines.py single  --help
Usage: baselines.py single [OPTIONS] BASELINE [DATASET]...

  Runs BASELINE on (a list of) DATASETs (See `list_baselines` for a list of
  available baselines).

Options:
  --examples / --no-examples  Whether or not to print example predictions
                              (default: false).
  --use-lemma / --no-lemma    Whether to make predictions on top of `Lemma`
                              baseline or not. (default: use it)
  --rounding / --no-rounding  Whether or not results should be rounded
                              (default: true).
  -t, --threshold FLOAT       Threshold for tunable baselines (default: 0.5);
                              ignored for non-tunable baselines.
  --help                      Show this message and exit.
```


## Evaluate all non-tunable baselines
To evaluate all non-tunable baselines on dev and test, run:
```
python3 code/baselines.py non_tunables data/dev.csv data/test.csv
```
The results will be stored to `non-tunable-dev.txt` and `non-tunable-test.txt`.

For more options, see `python3 code/baselines.py non_tunables --help`.


## Evaluate all tunable baselines
To tune all tunable baselines on dev and then evaluate on dev and test, run:
```
python3 code/baselines.py tunables data/dev.csv data/test.csv 
```
The results will be stored to `tunable-devtest.txt`.

For more options see `python3 code/baselines.py tunables --help`.


## Tune threshold of tunable baseline
To find the F1-optimal threshold for a single baseline on a given dataset (which should be dev, of course), run:
```
python3 code/baselines.py find_threshold DATASET BASELINE
```
Example:
```
python3 code/baselines.py find_threshold data/dev.csv typed_rel_emb
```


## Determine which type signatures benefit from types
For the baseline `w2v+tsg_rel_emb`, the effectiveness of type-informed vs. unrestricted (untyped) relation embeddings has to be determined before-hand.
For this run:
```
python3 code/baselines.py tsg_pref data/dev.csv tsg_typed_vs_untyped.txt
```
This will store the type signature preferences in the file `tsg_typed_vs_untyped.txt`.
To use this file, you have to enter the path to it in `file_paths.json`.

A precomputed file is available in `data/tsg_typed_vs_untyped_thr0.0-only-dev.txt`.
So `w2v+tsg_rel_emb` can be used right away without any preprocessing necessary (i.e., other than downloading the pretrained word2vec embeddings).


## Error Analysis
If you want to qualitatively analyze errors made by several tunable baselines on a specific dataset (which should be dev),
you can run
```
python3 code/baselines.py error_analysis DATASET OUT_FILE [METHODS]...
```
where results will be written to `OUT_FILE`. You can specify as many `METHODS` as you want.


## Convert an event graph to a word2vec training corpus
The method to convert the SherLIiC Event Graph to a training corpus suitable for word2vec is given by the command `create_training`:
```
python3 code/baselines.py create_training data/teg.tsv typed_rel_emb_train.txt
```
The command above would create the training corpus used for learning the embeddings in `embeddings/complete/typed_rel_emb.txt`, i.e., the embeddings for the baseline `typed_rel_emb`.

See `python3 code/baselines.py create_training --help` for more options.


## word2vec baseline
In order to use the word2vec baseline (and all baselines building on it), you have to

1. Download the pretrained word embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Enter the path to the gzipped file in `file_paths.json` under the key `word2vec`.


## Rule Collection Baselines

In order to reproduce the rule collection baselines, you have to download them,
sometimes run a preprocessing script and enter the path to the right file into `file_paths.json`. Find specific instructions below.

### Berant I
1. Download `ACL2011Resource.zip` from [here](https://www.cs.tau.ac.il/~joberant/resources.html).
2. Unzip it and enter the path to `ResourceEdges.txt` in `file_paths.json` under the key `berant`.

### Berant II
1. Download `reverb_local_global.rar` from [here](http://u.cs.biu.ac.il/~nlp/resources/downloads/predicative-entailment-rules-learned-using-local-and-global-algorithms/).
2. Extract the archive and enter the path to `reverb_local_clsf_all.txt` in `file_paths.json` under the key `berant_new`.

### PPDB
1. Download PPDB 2.0 XXXL All from [here](http://paraphrase.org/#/download).
2. Run `python3 code/preprocess.py ppdb path/to/ppdb-2.0-xxxl-all.gz external/ppdb.csv`
3. Enter the path `external/ppdb.csv` in `file_paths.json` under the key `ppdb`.

### Patty
1. Download `patty-dataset-freebase.tar.gz` from [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/patty/).
2. Extract the archive and enter the paths to `wikipedia-patterns.txt` and `wikipedia-subsumptions.txt` as two-element list in `file_paths.json` under the key `patty`.

### Schoenmackers
1. Download `sherlockrules.zip` from [here](http://projectsweb.cs.washington.edu/research/sherlock-hornclauses/).
2. Extract the archive, enter the folder `sherlockrules` and run
```
cat sherlockrules.* | grep -v '^#' | cut -f1,2,9 | grep -v '2\.0' | cut -f1,3 > sherlockrules.all
```
3. Enter the path to `sherlockrules.all` in `file_paths.json` under the key `schoenmackers`.

### Chirps
1. Download `resource.zip` from [here](https://github.com/vered1986/Chirps/tree/master/resource).
2. Unzip it and enter the path to `rules.tsv` in `file_paths.json` under the key `chirps`.

### All Rules
Once you have downloaded all resources and put the right paths into `file_paths.json`,
you should be able to run this baseline, too.
