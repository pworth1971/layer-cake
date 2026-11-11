"""Figures for the revised manuscript (vector PDF, sized for the 32pc text block).

Reads ../data/runs.csv and ../data/derived/*.csv written by analyze.py.
Colour encodes classifier family in a fixed order (validated all-pairs for CVD with
the dataviz palette validator); marker shape repeats the family so identity never
depends on colour alone, and every figure has a table twin in the manuscript.

Usage: python figures.py  (run from the analysis/ directory, after analyze.py)
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DERIVED = DATA / "derived"
OUT = HERE.parent / "pictures"

TEXTWIDTH = 5.0  # inches: the 12.71 cm IJIS body column
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
FAM_COLOR = {"ML": "#2a78d6", "static": "#eb6834", "transformer": "#1baf7a"}
FAM_MARKER = {"ML": "o", "static": "s", "transformer": "^"}
FAM_LABEL = {"ML": "Classical ML", "static": "Static-embedding NN", "transformer": "Fine-tuned transformer"}
HIGHLIGHT, CONTEXT = "#eb6834", "#2a78d6"

DATASETS = ["bbc-news", "20newsgroups", "imdb", "arxiv_protoformer",
            "reuters21578", "ohsumed", "arxiv", "rcv1"]
DS_LABEL = {"bbc-news": "BBC News", "20newsgroups": "20 Newsgroups", "imdb": "IMDB",
            "arxiv_protoformer": "ArXiv-10", "reuters21578": "Reuters-21578",
            "ohsumed": "OHSUMED", "arxiv": "arXiv", "rcv1": "RCV1-v2"}
SINGLE = DATASETS[:4]
LM = {"llama": "Llama-3.2-1B", "deepseek": "DeepSeek-R1-Distill", "gpt2": "GPT-2", "roberta": "RoBERTa",
      "xlnet": "XLNet", "bert": "BERT", "distilbert": "DistilBERT"}

mpl.rcParams.update({
    # the IJIS template asks for 8 pt Times New Roman figure labels
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
})


def family_handles(families=("ML", "static", "transformer")):
    return [Line2D([], [], ls="", marker=FAM_MARKER[f], ms=5.5, mfc=FAM_COLOR[f], mec=SURFACE, mew=0.8,
                   label=FAM_LABEL[f]) for f in families]


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def log_ticks(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, numticks=8))
    a.set_minor_formatter(NullFormatter())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))


def fig_peak_gaps():
    """How far the best classical-ML and best static-NN runs sit behind the best transformer run."""
    peak = pd.read_csv(DERIVED / "peak.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), sharey=True)
    y = np.arange(len(DATASETS))[::-1]
    for ax, measure, title in zip(axes, ["macro", "micro"], ["Macro-F1", "Micro-F1"]):
        p = peak[peak.measure == measure].pivot(index="dataset", columns="family", values="value").loc[DATASETS]
        gap = {f: 100 * (p["transformer"] - p[f]) for f in ["ML", "static"]}
        for yi, ds in zip(y, DATASETS):
            lo, hi = min(0, gap["ML"][ds], gap["static"][ds]), max(gap["ML"][ds], gap["static"][ds])
            ax.plot([lo, hi], [yi, yi], color=GRID, lw=1.6, solid_capstyle="round", zorder=1)
        for f, dy in [("ML", 0.13), ("static", -0.13)]:
            ax.scatter(gap[f].values, y + dy, s=26, marker=FAM_MARKER[f], color=FAM_COLOR[f],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.axvline(0, color=FAM_COLOR["transformer"], lw=1.2, zorder=2)
        ax.axhline(3.5, color=AXIS, lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("F1 points behind the best transformer")
        ax.set_xlim(-1.5, 18)
        ax.grid(axis="y", visible=False)
        ax.text(17.5, 7.25, "single-label", ha="right", va="center", fontsize=6.5, color=MUTED)
        ax.text(17.5, 3.25, "multi-label", ha="right", va="center", fontsize=6.5, color=MUTED)
    axes[0].set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    handles = family_handles(("ML", "static")) + [Line2D([], [], color=FAM_COLOR["transformer"], lw=1.2,
                                                         label="Best transformer (reference)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_peak_gaps")


def fig_pareto():
    """Macro-F1 against training time for every run, with the Pareto frontier."""
    runs = pd.read_csv(DATA / "runs.csv")
    runs = runs[runs.measure == "macro"]
    front = pd.read_csv(DERIVED / "pareto.csv")
    front = front[front.measure == "macro"]
    style = {"ML": dict(s=4, alpha=0.35, lw=0), "static": dict(s=11, alpha=0.8, lw=0.4),
             "transformer": dict(s=13, alpha=0.85, lw=0.4)}
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.5))
    for ax, ds in zip(axes.flat, DATASETS):
        g = runs[runs.dataset == ds]
        f = front[front.dataset == ds].sort_values("time_s")
        xs = np.repeat(f.time_s.values, 2)[1:]
        ys = np.repeat(f.value.values, 2)[:-1]
        for fam in ["ML", "static", "transformer"]:
            h = g[g.family == fam]
            ax.scatter(h.time_s, h.value, marker=FAM_MARKER[fam], color=FAM_COLOR[fam], edgecolor=SURFACE,
                       zorder=2 if fam == "ML" else 3, **style[fam])
        ax.plot(xs, ys, color=INK, lw=0.9, zorder=4)
        ax.set_xscale("log")
        log_ticks(ax)
        hi = g.value.max()
        lo = max(g.value.quantile(0.05), hi - 0.25)
        ax.set_ylim(lo - 0.01, hi + 0.012)
        ax.set_title(DS_LABEL[ds], fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    for ax in axes[1]:
        ax.set_xlabel("Training time (s, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    handles = family_handles() + [Line2D([], [], color=INK, lw=0.9, label="Pareto frontier")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=0.9, w_pad=0.5)
    save(fig, "fig_pareto")


def fig_cd():
    """Critical-difference diagrams (Friedman + Nemenyi) over median F1 across eight datasets."""
    ranks = pd.read_csv(DERIVED / "ranks.csv")
    res = json.loads((HERE / "results.json").read_text())["friedman"]
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.2))
    for ax, measure, title in zip(axes, ["macro", "micro"], ["Median Macro-F1", "Median Micro-F1"]):
        r = ranks[ranks.measure == measure].sort_values("avg_rank").reset_index(drop=True)
        cd, k = res[measure]["cd"], res[measure]["k"]
        ax.set_xlim(0.4, k + 0.6)
        ax.set_ylim(-1.0, 1.0)
        ax.axis("off")
        ax.plot([1, k], [0.55, 0.55], color=INK2, lw=0.8)
        for t in range(1, k + 1):
            ax.plot([t, t], [0.55, 0.62], color=INK2, lw=0.8)
            ax.text(t, 0.68, str(t), ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.plot([k - cd, k], [0.92, 0.92], color=INK, lw=1.0)
        for e in (k - cd, k):
            ax.plot([e, e], [0.87, 0.97], color=INK, lw=1.0)
        ax.text(k - cd / 2, 0.98, f"critical difference = {cd:.2f}", ha="center", va="bottom", fontsize=6.5)
        vals = r.avg_rank.values
        cliques = []
        for i in range(len(vals)):
            j = i
            while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in cliques):
                cliques.append((i, j))
        for n, (a, b) in enumerate(cliques):
            yy = 0.42 - 0.09 * n
            ax.plot([vals[a] - 0.04, vals[b] + 0.04], [yy, yy], color=INK, lw=2.2, solid_capstyle="round")
        half = (len(r) + 1) // 2
        for i, row in r.iterrows():
            left = i < half
            yy = -0.05 - 0.21 * (i if left else len(r) - 1 - i)
            xx = 0.55 if left else k + 0.45
            ax.plot([row.avg_rank, row.avg_rank, xx], [0.55, yy, yy], color=MUTED, lw=0.6)
            ax.scatter([row.avg_rank], [0.55], s=22, marker=FAM_MARKER[row.family], color=FAM_COLOR[row.family],
                       edgecolor=SURFACE, linewidth=0.7, zorder=4)
            ax.text(xx + (-0.05 if left else 0.05), yy, f"{row.classifier} ({row.avg_rank:.2f})",
                    ha="right" if left else "left", va="center", fontsize=6.8)
        ax.set_title(title, pad=2)
    fig.legend(handles=family_handles(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=0.2)
    save(fig, "fig_cd")


def fig_solo():
    """Pooled language-model embeddings alone ('solo') vs. the best classical representation form."""
    f = pd.read_csv(DERIVED / "solo.csv")
    f = f[f.measure == "macro"].set_index("dataset").loc[DATASETS]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.15))
    y = np.arange(len(f))[::-1]
    for yi, (ds, r) in zip(y, f.iterrows()):
        ax.plot([r.solo_time, r.best_time], [yi, yi], color=GRID, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter([r.best_time], [yi], s=26, marker="D", color=CONTEXT, edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.scatter([r.solo_time], [yi], s=26, marker="o", color=HIGHLIGHT, edgecolor=SURFACE, lw=0.8, zorder=4)
        ax.text(1.02, yi, f"+{r.gap_pts:.1f} pts at {r.time_ratio:.1f}$\\times$ the time ({r.best_form})",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=INK2)
    ax.axhline(3.5, color=AXIS, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1, 2e4)
    ax.set_xticks([1, 10, 60, 600, 3600], ["1 s", "10 s", "1 min", "10 min", "1 h"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, [DS_LABEL[d] for d in f.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median training time of the representation form (log scale)")
    handles = [Line2D([], [], ls="", marker="o", ms=5.5, mfc=HIGHLIGHT, mec=SURFACE, label="solo (pooled embeddings only)"),
               Line2D([], [], ls="", marker="D", ms=5, mfc=CONTEXT, mec=SURFACE, label="best representation form")]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.75, 1.2))
    fig.subplots_adjust(left=0.2, right=0.6, bottom=0.2, top=0.87)
    save(fig, "fig_solo")


def fig_lms():
    """Fine-tuned transformer models: distance from the best model per dataset, and training time."""
    long = pd.read_csv(DERIVED / "lm_long.csv")
    order = pd.read_csv(DERIVED / "lm_table.csv").lm.tolist()
    bert_t = long[long.lm == "bert"].set_index("dataset").time_med
    long["rel_time"] = long.time_med / long.dataset.map(bert_t)
    col = FAM_COLOR["transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    y = {lm: i for i, lm in enumerate(order[::-1])}
    rng = np.random.default_rng(7)
    for ax, xcol in zip(axes, ["behind_pts", "rel_time"]):
        for lm in order:
            g = long[long.lm == lm]
            jit = rng.uniform(-0.18, 0.18, len(g))
            single = g.dataset.isin(SINGLE).values
            ax.scatter(g[xcol][single], y[lm] + jit[single], s=16, marker="^", color=col, edgecolor=SURFACE, lw=0.5, zorder=3)
            ax.scatter(g[xcol][~single], y[lm] + jit[~single], s=16, marker="^", facecolor=SURFACE, edgecolor=col, lw=0.9, zorder=3)
            med = g[xcol].median()
            ax.plot([med, med], [y[lm] - 0.32, y[lm] + 0.32], color=INK, lw=1.4, zorder=4)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Points behind the best model (Macro-F1)")
    axes[0].set_xlim(-0.5, 10.8)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].axvline(1, color=AXIS, lw=0.8)
    axes[1].set_xlabel("Training time vs. BERT (median, log)")
    axes[0].set_yticks(list(y.values()), [LM[k] for k in y])
    handles = [Line2D([], [], ls="", marker="^", ms=5, mfc=col, mec=SURFACE, label="single-label corpus"),
               Line2D([], [], ls="", marker="^", ms=5, mfc=SURFACE, mec=col, mew=0.9, label="multi-label corpus"),
               Line2D([], [], color=INK, lw=1.4, label="median over corpora")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_lms")


def fig_class_embeddings():
    """Paired effect of class embeddings: best run with vs. without, same dataset, model and classifier."""
    pairs = pd.read_csv(DERIVED / "class_embedding_pairs.csv")
    pairs = pairs[pairs.measure == "macro"]
    rows = [("TCE, transformer", None, "TCE in fine-tuned transformers", "transformer"),
            ("WCE, static NN", None, "WCE in static-embedding NNs", "static"),
            ("WCE, classical ML", "cat-doc", "WCE in classical ML: cat-doc", "ML"),
            ("WCE, classical ML", "lsa", "WCE in classical ML: lsa", "ML"),
            ("WCE, classical ML", "solo", "WCE in classical ML: solo", "ML"),
            ("WCE, classical ML", "dot", "WCE in classical ML: dot", "ML")]
    lo, hi = -12, 26
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.3))
    rng = np.random.default_rng(3)
    labels = []
    for i, (kind, base, label, fam) in enumerate(rows):
        d = pairs[pairs.kind == kind]
        if base:
            d = d[d.base == base]
        v = d.delta_pts.values
        inside = v[(v >= lo) & (v <= hi)]
        jit = rng.uniform(-0.22, 0.22, len(inside))
        ax.scatter(inside, i + jit, s=7, marker=FAM_MARKER[fam], color=FAM_COLOR[fam], alpha=0.55, lw=0, zorder=2)
        med = np.median(v)
        ax.plot([med, med], [i - 0.33, i + 0.33], color=INK, lw=1.6, zorder=4)
        out_hi = v[v > hi]
        if len(out_hi):
            # points beyond the axis are counted in the margin rather than drawn
            ax.text(1.01, i, f"{len(out_hi)} more, up to +{out_hi.max():.0f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.3, color=INK2)
        labels.append(f"{label} (n = {len(v)})")
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(rows)), labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in best Macro-F1 when class embeddings are added (points)")
    handles = family_handles() + [Line2D([], [], color=INK, lw=1.6, label="median")]
    ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.45, 1.2))
    fig.subplots_adjust(left=0.36, right=0.83, bottom=0.2, top=0.86)
    save(fig, "fig_classemb")


if __name__ == "__main__":
    for p in OUT.glob("fig_*"):
        p.unlink()
    fig_peak_gaps()
    fig_pareto()
    fig_cd()
    fig_solo()
    fig_lms()
    fig_class_embeddings()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.pdf")))
