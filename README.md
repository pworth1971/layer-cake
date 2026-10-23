# Layer Cake

Layer Cake is a benchmarking framework for text classification that measures **effectiveness and training cost on the same runs**. It trains three families of classifiers over eight single- and multi-label corpora, logging Macro-/Micro-F1, training time and hardware for every run:

- linear models over sparse, latent-semantic and pooled-embedding features;
- neural networks over static word embeddings;
- fine-tuned transformers.

It accompanies the paper *Layer Cake: On Language Representation and Compute Characteristics in Text Classification* (Worth & Cardei, [International Journal of Intelligence Science 15(4), 2025](https://doi.org/10.4236/ijis.2025.154010)). A revised and extended version is in preparation; its source, data and analysis scripts are in [`paper/scirp/V7/`](paper/scirp/V7/).

## Key findings

These come from the revised analysis of 3,725 runs:

- **Peak accuracy.** The best fine-tuned transformer beats the best linear configuration by a mean of **3.2 Macro-F1 points** (range 0.1–7.0) and the best static-embedding network by **6.7 points**.
- **Training cost.** A typical transformer run trains **75×** longer than a typical linear run and **32×** longer than a typical static-embedding run (**127×** the GPU-time).
- **Task dependence.** On single-label corpora, a linear model within 0.5–2.7 points of the best transformer trains in under 0.5% of its time. On multi-label, many-class corpora, the transformer's 3.7–7.0-point lead has no linear substitute.
- **Linear models vs. static networks.** Linear models over frozen, mean-pooled transformer embeddings beat every static-embedding network.
- **Model size and class embeddings.** Billion-parameter-scale decoders (Llama-3.2-1B, DeepSeek-R1-Distill-Qwen-1.5B) cost about 5× as much to fine-tune as base encoders, for no consistent gain. Word-class embeddings help static-embedding networks; their token-level analogue has no effect on fine-tuned transformers.

## Modules

All programs live in `src/` and are run from `bin/`. Relative paths such as `../log` and `../.vector_cache` assume that working directory.

| Module | Program | Classifiers | Text representation | Hardware |
|---|---|---|---|---|
| Classical ML | `ml_classification_test_v5.0.py` | logistic regression, linear SVM, multinomial naive Bayes (scikit-learn; one-vs-rest for multi-label) | TF-IDF or counts, LSA, pooled static or frozen transformer embeddings, word-class embeddings, and combinations | CPU |
| Static-embedding NN | `wrd_layer_cake_v3.0.py` | CNN, LSTM, attention (PyTorch) | GloVe, word2vec or fastText vectors over the corpus vocabulary, frozen or fine-tuned, optionally extended with word-class embeddings | 1 GPU (CUDA or Apple MPS) |
| Transformer | `trans_layer_cake_v13.5.py` | Hugging Face sequence classifier (`hf.sc`) or a CNN head (`hf.cnn`), all weights fine-tuned | BERT, RoBERTa, DistilBERT, XLNet, GPT-2, Llama-3.2-1B, DeepSeek-R1-Distill-Qwen-1.5B; optional token-class embeddings | all visible GPUs (PyTorch DataParallel) |
| Analysis | `results_analysis_v4.0.py` | — | summaries, charts and heatmaps from run logs | CPU |

The static-embedding module and the word-class embedding (WCE) code are derived from the code of Moreo, Esuli and Sebastiani, [*Word-class embeddings for multiclass text classification*](https://doi.org/10.1007/s10618-020-00735-3) (Data Mining and Knowledge Discovery 35, 2021; [original repository](https://github.com/AlexMoreo/word-class-embeddings)).

## Repository layout

```text
bin/            driver scripts: environment setup, experiment sweeps, analysis
src/            programs (see Modules) and their packages:
  data/           dataset loaders and readers
  embedding/      word-class / token-class embedding computation
  model/          language-model wrappers (LCRepresentationModel) and classifiers
  util/           shared settings, logging, metrics
  test/, deprecated/   earlier versions, kept for reference
paper/          manuscripts and submissions; paper/scirp/V7 is the current revision
datasets/       corpora (not tracked; see Datasets)
.vector_cache/  pretrained embeddings and model downloads (not tracked)
pickles/        cached, preprocessed datasets (not tracked)
log/            run logs (not tracked)
out/            analysis output (not tracked)
```

## Installation

The code was developed and run on Ubuntu 22.04 with CUDA 12.4 and on macOS (Apple Silicon, MPS), using Python 3.10.

```bash
cd bin
bash setup.sh
```

`setup.sh` does the following:

- creates a conda environment named `python310`;
- installs PyTorch 2.4.1 (CUDA build on Linux), Transformers, scikit-learn, gensim and the other dependencies in `requirements.txt`;
- asks you to log in to Hugging Face, because Llama models are gated;
- creates the `datasets/`, `.vector_cache/`, `pickles/`, `log/` and `out/` directories.

> **Before running `setup.sh`:** its last lines set the *global* git `user.name` and `user.email` to the original author's. Delete or edit them first.

DeepSeek models were not supported on macOS in our setup; run them on Linux.

## Datasets

Pass the key with `--dataset`. Corpora marked *automatic* are downloaded on first use; the others must be placed by hand at the path shown. The first load of each corpus is cached in `pickles/`.

| Corpus | Key | Labels | Classes | Location | Source |
|---|---|---|---|---|---|
| BBC News | `bbc-news` | single | 5 | `datasets/bbc-news/BBC News Train.csv` | Kaggle "BBC News Classification" competition (1,490 labelled articles) |
| 20 Newsgroups | `20newsgroups` | single | 20 | scikit-learn cache | automatic: `fetch_20newsgroups`, by-date split, headers/footers/quotes removed |
| IMDB | `imdb` | single | 2 | `datasets/imdb/` | automatic: Hugging Face `datasets` |
| ArXiv-10 | `arxiv_protoformer` | single | 10 | `datasets/arxiv_protoformer/arxiv100.csv` | ArXiv-10 release of Protoformer (Farhangi et al., PAKDD 2022) |
| Reuters-21578 | `reuters21578` | multi | 115 | `datasets/reuters21578/` | automatic: UCI archive, ModApté split |
| OHSUMED | `ohsumed` | multi | 23 | `datasets/ohsumed50k/` | automatic: `ohsumed-all-docs.tar.gz` (23 MeSH disease categories) |
| arXiv | `arxiv` | multi | 58 | `datasets/arxiv/arxiv-metadata-oai-snapshot.json` | Kaggle "arXiv Dataset" (Cornell University metadata snapshot) |
| RCV1-v2 | `rcv1` | multi | 101 | `datasets/rcv1/` | the RCV1 XML distribution, obtained from NIST under the Reuters research agreement; the LYRL2004 split and topic labels are downloaded automatically |

Two path details:

- OHSUMED is read from `datasets/ohsumed50k/`, not the `datasets/ohsumed/` folder that `setup.sh` creates.
- On Linux, directory names are case-sensitive (`datasets/imdb`).

## Language models

| Key (`--pretrained`) | Checkpoint | Dim. | Notes |
|---|---|---|---|
| `glove` | `glove.840B.300d` | 300 | downloaded from Stanford NLP on first use |
| `word2vec` | `GoogleNews-vectors-negative300.bin` | 300 | downloaded on first use; if the Google Drive link fails, place the file in `.vector_cache/Word2Vec/` (also on Kaggle) |
| `fasttext` | `crawl-300d-2M.vec` | 300 | word-level Common Crawl vectors (no subword model), converted to binary on first use |
| `bert` | `bert-base-uncased` (transformer module); `bert-base-cased` (classical module default) | 768 | Hugging Face |
| `distilbert` | `distilbert-base-cased` / `-uncased` | 768 | Hugging Face |
| `roberta` | `roberta-base` | 768 | Hugging Face |
| `xlnet` | `xlnet-base-cased` | 768 | Hugging Face |
| `gpt2` | `gpt2` | 768 | Hugging Face |
| `llama` | `meta-llama/Llama-3.2-1B` | 2048 | Hugging Face, gated: accept Meta's license and log in |
| `deepseek` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1536 | Hugging Face |

Embeddings and model weights are cached under `.vector_cache/<Model>/`. Checkpoint names are set in `src/model/LCRepresentationModel.py`.

## Running experiments

Run everything from `bin/`. Each program accepts `--help`. Results are appended to the log file named by `--logfile` (classical module) or `--log-file` (neural modules). A run whose configuration is already in the log is skipped unless `--force` is given.

### Classical ML

```bash
python ../src/ml_classification_test_v5.0.py --dataset 20newsgroups --net lr \
    --vtype tfidf --pretrained llama --mix cat-doc --dataset-emb-comp avg \
    --logfile ../log/ml_test.test
```

Main options:

- `--net`: `svm`, `lr` or `nb`.
- `--vtype`: `tfidf` or `count`.
- `--pretrained`: any key from the language-model table.
- `--dataset-emb-comp`: document pooling; `avg` (mean), `summary` (single summary token) or `weighted` (TF-IDF-weighted mean).
- `--optimc`: randomized hyperparameter search. Off by default; the published runs use default hyperparameters.

`--mix` selects the feature set. Here *x* is the TF-IDF (or count) vector, *ē* the pooled document embedding, *E* the model's vocabulary embedding matrix and *W* the word-class embedding matrix:

| `--mix` | Features |
|---|---|
| `vmode` | *x* |
| `lsa` | truncated SVD of *x*, to the embedding dimension |
| `solo` | *ē* |
| `dot` | *x · E* |
| `cat-doc` | *x* ⊕ *ē* |
| `lsa-wce` | SVD(*x*) ⊕ *x · [E ⊕ W]* |
| `solo-wce` | *ē* ⊕ *x · [E ⊕ W]* |
| `dot-wce` | *x · [E ⊕ W]* |
| `cat-wce` | *x* ⊕ *x · [E ⊕ W]* |
| `cat-doc-wce` | *x* ⊕ *ē* ⊕ *x · [E ⊕ W]* |

Naive Bayes needs non-negative inputs, so use it with `vmode` only.

### Static-embedding neural networks

```bash
python ../src/wrd_layer_cake_v3.0.py --dataset reuters21578 --net cnn --channels 128 \
    --pretrained glove --supervised --tunable --nepochs 77 --patience 5 \
    --log-file ../log/nn_test.test
```

Main options:

- `--net`: `cnn`, `lstm` or `attn`.
- `--pretrained`: `glove`, `word2vec` or `fasttext`.
- `--tunable`: fine-tune the pretrained vectors.
- `--supervised`: append word-class embeddings. `--supervised-method` chooses how they are computed: `ig` (default), `dotn`, `ppmi` or `chi`.
- `--channels` / `--hidden`: CNN width and LSTM/attention width.
- Training: `--nepochs`, `--patience`, `--lr`, `--batch-size`.

Early stopping monitors validation Macro-F1.

### Fine-tuned transformers

```bash
python ../src/trans_layer_cake_v13.5.py --dataset imdb --pretrained roberta \
    --net hf.sc --tunable --epochs 33 --patience 5 \
    --log-file ../log/trans_test.test
```

Main options:

- `--net`: `hf.sc` for the Hugging Face sequence-classification head, `hf.cnn` for the CNN head.
- `--pretrained`: `bert`, `roberta`, `distilbert`, `xlnet`, `gpt2`, `llama` or `deepseek`.
- `--tunable`: fine-tune all weights. The published runs use it throughout.
- `--supervised`: fuse token-class embeddings with the pooled encoder output, using `--sup-mode` `cat` (default), `add` or `dot`.

The model is replicated across all visible GPUs with DataParallel; restrict them with `CUDA_VISIBLE_DEVICES`. Llama and DeepSeek need GPUs with enough memory to hold the full model on each device.

### Sweeps

The scripts in `bin/` run grids of the commands above. Edit the arrays at the top of each script (datasets, models, representations, log file) to set the sweep:

| Script | Runs |
|---|---|
| `ml_testing.sh` | classical module |
| `lc_nn_word_test.sh` | static-embedding networks |
| `trans_lc_hfseq_test.sh` | transformers with the sequence-classification head |
| `trans_lc_hfcnn_test.sh` | transformers with the CNN head |
| `analyze_ml.sh`, `analyze_nn.sh` | the analysis program on the classical and neural logs |

## Logs and analysis

Each run appends one tab-separated row per metric to its log. Each row records:

- the corpus, classifier, language model and representation;
- the feature dimensions and hyperparameters;
- the metric and its value: Macro- and Micro-F1, precision, recall, accuracy, Hamming loss, Jaccard index;
- the training time (`timelapse`, seconds; excludes data preparation, embedding extraction and inference);
- the epoch, OS, CPU, GPU and memory.

```bash
python ../src/results_analysis_v4.0.py ../log/ml_test.test --summary --charts --heatmaps --analysis
```

This writes text and CSV summaries, per-corpus result files, charts and heatmaps to `out/<log-file-name>/`.

The analysis behind the revised paper (peak and typical scores, significance tests, Pareto frontiers, tolerance tables, model-size and class-embedding comparisons) is in [`paper/scirp/V7/analysis/`](paper/scirp/V7/analysis/). It works from the run-level results in `paper/scirp/V7/data/runs/` and regenerates every table and figure in that manuscript; see [`paper/scirp/V7/REVISION_NOTES.md`](paper/scirp/V7/REVISION_NOTES.md).

## Citation

```bibtex
@article{worth2025layercake,
  title   = {Layer Cake: On Language Representation and Compute Characteristics in Text Classification},
  author  = {Worth, Jr., Peter J. and Cardei, Ionut},
  journal = {International Journal of Intelligence Science},
  volume  = {15},
  number  = {4},
  pages   = {184--262},
  year    = {2025},
  doi     = {10.4236/ijis.2025.154010}
}
```

If you use the word-class embedding code, please also cite:

```bibtex
@article{moreo2021,
  title   = {Word-class embeddings for multiclass text classification},
  author  = {Moreo, Alejandro and Esuli, Andrea and Sebastiani, Fabrizio},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {35},
  number  = {3},
  pages   = {911--963},
  year    = {2021},
  doi     = {10.1007/s10618-020-00735-3}
}
```

## License

BSD 3-Clause; see [LICENSE](LICENSE), inherited from the word-class embeddings code base this project extends. Questions: <pworth2022@fau.edu>.
