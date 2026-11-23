"""Write the manuscript's results tables as LaTeX fragments (../tables/*.tex).

Every number is read from ../data/runs.csv or ../data/derived (analyze.py); nothing is
typed by hand.

Usage: python tables.py  (run from the analysis/ directory, after analyze.py)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DERIVED = DATA / "derived"
OUT = HERE.parent / "tables"

DATASETS = ["bbc-news", "20newsgroups", "imdb", "arxiv_protoformer",
            "reuters21578", "ohsumed", "arxiv", "rcv1"]
DS_LABEL = {"bbc-news": "BBC News", "20newsgroups": "20 Newsgroups", "imdb": "IMDB",
            "arxiv_protoformer": "ArXiv-10", "reuters21578": "Reuters-21578",
            "ohsumed": "OHSUMED", "arxiv": "arXiv", "rcv1": "RCV1-v2"}
LM = {"llama": "Llama", "deepseek": "DeepSeek", "gpt2": "GPT-2", "roberta": "RoBERTa", "xlnet": "XLNet",
      "glove": "GloVe", "fasttext": "fastText", "word2vec": "word2vec", "bert": "BERT", "distilbert": "DistilBERT"}
CLF = {"lr": "LR", "svm": "SVM", "nb": "NB", "cnn": "CNN", "lstm": "LSTM", "attn": "ATTN",
       "hf.sc": "SC", "hf.cnn": "CNN"}
CLF_LONG = dict(CLF, **{"hf.sc": "HF-SC", "hf.cnn": "HF-CNN"})


def split_top(s, sep="+"):
    """Split on separators that are not inside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def rep_label(r):
    """Compact notation for an ML feature set; see the notation paragraph in Section 3.3."""
    if not isinstance(r, str) or not r:
        return ""
    if r.startswith("tfidf->LSA") or r.startswith("count->LSA"):
        return "LSA"
    if re.fullmatch(r"(tfidf|count)[ \[]\w+\]?", r):
        return "TF-IDF" if r.startswith("tfidf") else "Count"
    parts = []
    tokens = split_top(r)
    for i, p in enumerate(tokens):
        m = re.fullmatch(r"(?:tfidf|count)\.\((\w+)\+wce\((?:tfidf|count)\)\)", p)
        if m and p in tokens[i + 1:]:
            # the logger labels cat-doc-wce as tfidf+P+P; the first block is the pooled embedding
            parts.append(LM[m.group(1)])
        elif m:
            parts.append(f"$\\Pi$({LM[m.group(1)]}+WCE)")
        elif p in ("tfidf", "count"):
            parts.append("TF-IDF" if p == "tfidf" else "Count")
        else:
            m = re.fullmatch(r"(\w+)\((?:avg|weighted|summary)\)", p)
            parts.append(LM[m.group(1)] if m else p)
    return " $\\oplus$ ".join(parts)


def secs(x):
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def pct(x):
    if x < 0.0001:
        return "$<$0.01\\%"
    return f"{100 * x:.0f}\\%" if x >= 0.095 else f"{100 * x:.1f}\\%" if x >= 0.0095 else f"{100 * x:.2f}\\%"


def model_label(row):
    """Who produced a run: 'LR', 'CNN' or, for transformers, 'RoBERTa/SC'."""
    if row.family == "transformer":
        return f"{LM[row.lm]}/{CLF[row.classifier]}"
    return CLF[row.classifier]


def write(name, body):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def group_rule(i, ncols, label=None):
    if i != 4:
        return []
    rows = ["\\midrule"]
    if label:
        rows.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
    return rows


def table_peak(measure):
    peak = pd.read_csv(DERIVED / "peak.csv")
    p = peak[peak.measure == measure].set_index(["dataset", "family"], drop=False)
    lines = []
    for i, ds in enumerate(DATASETS):
        lines += group_rule(i, 12)
        ml, st, tr = (p.loc[(ds, f)] for f in ["ML", "static", "transformer"])
        cells = [DS_LABEL[ds]]
        for r in (ml, st, tr):
            cells += [f"{r.value:.3f}", model_label(r), secs(r.time_s)]
        cells += [f"{100 * (tr.value - ml.value):+.1f}", f"{100 * (tr.value - st.value):+.1f}"]
        lines.append(" & ".join(cells) + " \\\\")
    means = peak[peak.measure == measure].groupby("family").value.mean()
    lines.append("\\midrule")
    lines.append(f"Mean & {means['ML']:.3f} & & & {means['static']:.3f} & & & {means['transformer']:.3f} & & & "
                 f"{100 * (means['transformer'] - means['ML']):+.1f} & {100 * (means['transformer'] - means['static']):+.1f} \\\\")
    write(f"tab_peak_{measure}", "\n".join(lines) + "\n")


def table_ml_configs(measure):
    peak = pd.read_csv(DERIVED / "peak.csv")
    p = peak[(peak.measure == measure) & (peak.family == "ML")].set_index("dataset")
    lines = [f"{DS_LABEL[ds]} & {CLF[p.loc[ds].classifier]} & {rep_label(p.loc[ds].config)} & {p.loc[ds].value:.3f} \\\\"
             for ds in DATASETS]
    write(f"tab_mlconfigs_{measure}", "\n".join(lines) + "\n")


def table_tests():
    w = pd.read_csv(DERIVED / "wilcoxon.csv")
    res = json.loads((HERE / "results.json").read_text())
    name = {("transformer", "ML"): "Transformer vs.\\ classical ML",
            ("transformer", "static"): "Transformer vs.\\ static NN",
            ("ML", "static"): "Classical ML vs.\\ static NN"}
    typ_key = {("transformer", "ML"): "T_minus_ML_pts", ("transformer", "static"): "T_minus_S_pts",
               ("ML", "static"): "ML_minus_S_pts"}
    lines = []
    for measure in ["macro", "micro"]:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{measure.capitalize()}-F1}}}} \\\\")
        for _, r in w[w.measure == measure].iterrows():
            k = (r.a, r.b)
            lines.append(f"\\quad {name[k]} & {r.mean_diff_pts:+.1f} & [{r.min_diff_pts:+.1f}, {r.max_diff_pts:+.1f}] & "
                         f"{r.wins}/{r.ties}/{r.losses} & {r.p:.4f} & {res['typical_gaps'][measure][typ_key[k]]:+.1f} \\\\")
    write("tab_tests", "\n".join(lines) + "\n")


def table_cost():
    runs = pd.read_csv(DATA / "runs.csv")
    runs = runs[runs.measure == "macro"]
    med = runs.groupby(["dataset", "family"]).time_s.median().unstack()
    c = pd.read_csv(DERIVED / "cost_ratios.csv")
    c = c[c.measure == "macro"].set_index("dataset")
    lines, ts, tm = [], [], []
    for i, ds in enumerate(DATASETS):
        lines += group_rule(i, 8)
        m = med.loc[ds]
        ts.append(m.transformer / m.static)
        tm.append(m.transformer / m.ML)
        lines.append(f"{DS_LABEL[ds]} & {c.loc[ds].gain_T_vs_ML_pts:+.1f} & {c.loc[ds].gain_T_vs_S_pts:+.1f} & "
                     f"{secs(m.ML)} & {secs(m.static)} & {secs(m.transformer)} & "
                     f"{ts[-1]:.0f}$\\times$ & {tm[-1]:.0f}$\\times$ \\\\")
    g = lambda x: float(np.exp(np.log(x).mean()))
    lines.append("\\midrule")
    lines.append(f"Geometric mean & & & & & & {g(ts):.0f}$\\times$ & {g(tm):.0f}$\\times$ \\\\")
    write("tab_cost", "\n".join(lines) + "\n")


def table_tolerance():
    t = pd.read_csv(DERIVED / "tolerance.csv")
    t = t[t.measure == "macro"]

    def who(row):
        if row.family != "ML":
            return model_label(row)
        # pure sparse or LSA features use the language model only as a tokenizer
        feats = {"vmode.tfidf": "TF-IDF", "vmode.count": "Count", "lsa": "LSA"}.get(row.form, LM.get(row.lm, row.lm))
        return f"{CLF[row.classifier]}, {feats}"

    lines = []
    for i, ds in enumerate(DATASETS):
        lines += group_rule(i, 6)
        if i:
            lines.append("\\addlinespace[2pt]")
        g = t[t.dataset == ds].set_index("eps_pts")
        b = g.loc[0.0]
        cells = [DS_LABEL[ds], f"{who(b)}\\newline {b.value:.3f}", secs(b.time_s)]
        for eps in (1.0, 3.0, 5.0):
            r = g.loc[eps]
            cells.append("---" if r.time_fraction == 1.0 and r.value == b.value else
                         f"{who(r)}\\newline {r.value:.3f} ({pct(r.time_fraction)})")
        lines.append(" & ".join(cells) + " \\\\")
    write("tab_tolerance", "\n".join(lines) + "\n")


def table_lms():
    t = pd.read_csv(DERIVED / "lm_table.csv")
    lines = [f"{LM[r.lm]} & {r.avg_rank:.2f} & {int(r.wins)} & {r.mean_behind_pts:.1f} & {r.max_behind_pts:.1f} & "
             f"{r.median_time_vs_bert:.2f}$\\times$ \\\\" for _, r in t.iterrows()]
    write("tab_lms", "\n".join(lines) + "\n")


def table_class_embeddings():
    res = json.loads((HERE / "results.json").read_text())["class_embeddings"]
    # in the classical module only dot-wce adds WCE columns alone; the other -wce forms add the
    # whole projection Pi_{M+WCE}, i.e. projected embedding features and WCEs together
    rows = [("WCE, static-embedding NNs", "WCE, static NN"),
            ("Classical, \\emph{dot} + WCE", ("WCE ML by base", "dot")),
            ("Classical, \\emph{solo} + $\\Pi_{M+\\mathrm{WCE}}$", ("WCE ML by base", "solo")),
            ("Classical, \\emph{lsa} + $\\Pi_{M+\\mathrm{WCE}}$", ("WCE ML by base", "lsa")),
            ("Classical, \\emph{cat-doc} + $\\Pi_{M+\\mathrm{WCE}}$", ("WCE ML by base", "cat-doc")),
            ("TCE, fine-tuned transformers", "TCE, transformer"),
            ("\\quad excluding collapsed runs", "TCE, transformer, no collapsed runs")]
    lines = []
    for label, key in rows:
        cells = [label]
        for measure in ["macro", "micro"]:
            s = res[f"{measure}|{key[0]}"][key[1]] if isinstance(key, tuple) else res[f"{measure}|{key}"]
            p = f"{s['p']:.3f}" if s["p"] < 0.1 else f"{s['p']:.2f}"
            med = "0.0" if abs(s["median"]) < 0.05 else f"{s['median']:+.1f}"
            cells += [med, f"{s['up']}/{s['down']}", f"{s['corpora_up']}/{s['corpora_down']}", p]
        cells.insert(1, str(s["n"]))
        cells.append(f"{s['time_ratio_median']:.1f}$\\times$")
        lines.append(" & ".join(cells) + " \\\\")
        if label.startswith("WCE, static") or "cat-doc" in label:
            lines.append("\\midrule")
    write("tab_classemb", "\n".join(lines) + "\n")


def table_hardware():
    h = pd.read_csv(DERIVED / "hardware.csv")
    fam_label = {"ML": "Classical ML", "static": "Static-embedding NN", "transformer": "Fine-tuned transformer"}
    machine_label = {"Linux workstation (CPU)": "Linux workstation, 64-core CPU",
                     "Apple M3 Max (CPU)": "Apple M3 Max laptop, 16-core CPU",
                     "NVIDIA RTX 6000 Ada Generation": "NVIDIA RTX 6000 Ada (48\\,GB)",
                     "NVIDIA A100-PCIE-40GB": "NVIDIA A100 PCIe (40\\,GB)",
                     "NVIDIA RTX A6000": "NVIDIA RTX A6000 (48\\,GB)"}
    lines = []
    for fam in ["ML", "static", "transformer"]:
        g = h[h.family == fam].groupby("machine").agg(runs=("runs", "sum"), datasets=("dataset", "nunique"))
        g = g.sort_values("runs", ascending=False)
        share = g.runs / g.runs.sum()
        acc = {"ML": "none", "static": "1 GPU", "transformer": "4 GPUs"}[fam]
        for j, (mach, r) in enumerate(g.iterrows()):
            first = j == 0
            lines.append(f"{fam_label[fam] if first else ''} & {machine_label[mach]} & {acc if first else ''} & "
                         f"{100 * share[mach]:.0f}\\% & {r.datasets} \\\\")
        if fam != "transformer":
            lines.append("\\midrule")
    write("tab_hardware", "\n".join(lines) + "\n")


def table_appendix(measure):
    runs = pd.read_csv(DATA / "runs.csv")
    r = runs[runs.measure == measure]
    agg = r.groupby(["dataset", "classifier"]).agg(max=("value", "max"), median=("value", "median"),
                                                    mean=("value", "mean"), min=("value", "min"),
                                                    n=("value", "size"), time=("time_s", "median"))
    order = ["lr", "svm", "nb", "cnn", "lstm", "attn", "hf.sc", "hf.cnn"]
    lines = []
    for i, ds in enumerate(DATASETS):
        if i:
            lines.append("\\midrule")
        for j, clf in enumerate(order):
            a = agg.loc[(ds, clf)]
            lines.append(f"{DS_LABEL[ds] if j == 0 else ''} & {CLF_LONG[clf]} & {a['max']:.3f} & {a['median']:.3f} & "
                         f"{a['mean']:.3f} & {a['min']:.3f} & {int(a.n)} & {secs(a.time)} \\\\")
    write(f"tab_appendix_{measure}", "\n".join(lines) + "\n")


def table_datasets():
    s = pd.read_csv(DATA / "dataset_splits.csv")
    domain = {"bbc-news": "news", "20newsgroups": "Usenet posts", "imdb": "movie reviews (sentiment)",
              "arxiv_protoformer": "paper abstracts", "reuters21578": "newswire", "ohsumed": "medical abstracts",
              "arxiv": "paper abstracts", "rcv1": "newswire"}
    ml = s[s.classifier_type == "ML"].set_index("dataset")
    lines = []
    for i, ds in enumerate(DATASETS):
        lines += group_rule(i, 6, "Multi-label")
        r = ml.loc[ds]
        lines.append(f"{DS_LABEL[ds]} & {domain[ds]} & {r.classes} & {r.docs:,} & {r.train:,} & {r.test:,} \\\\")
    write("tab_datasets", "\n".join(lines) + "\n")


def table_splits():
    s = pd.read_csv(DATA / "dataset_splits.csv")
    lines = []
    for i, ds in enumerate(DATASETS):
        lines += group_rule(i, 5)
        st = s[(s.dataset == ds) & s.classifier_type.str.startswith("DL (static")].iloc[0]
        tr = s[(s.dataset == ds) & s.classifier_type.str.startswith("DL (dynamic")].iloc[0]
        lines.append(f"{DS_LABEL[ds]} & {st.train:,} & {int(st.val):,} & {tr.train:,} & {int(tr.val):,} \\\\")
    write("tab_splits", "\n".join(lines) + "\n")


if __name__ == "__main__":
    for f in OUT.glob("*.tex"):
        f.unlink()
    for m in ["macro", "micro"]:
        table_peak(m)
        table_ml_configs(m)
        table_appendix(m)
    table_tests()
    table_cost()
    table_tolerance()
    table_lms()
    table_class_embeddings()
    table_hardware()
    table_datasets()
    table_splits()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
