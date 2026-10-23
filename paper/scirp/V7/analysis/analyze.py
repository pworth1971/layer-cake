"""Re-analysis for the revised manuscript, from run-level data.

Input: ../data/runs.csv (extract_runs.py), one row per run and measure. The V6 summary
tables in ../data/*.csv (extract_tables.py) are used only to verify that the run-level
data reproduce the published statistics.

Outputs: ../data/derived/*.csv and results.json, read by tables.py, figures.py and
quoted in the manuscript.

Definitions
-----------
family      ML = {lr, svm, nb}; static = {cnn, lstm, attn}; transformer = {hf.sc, hf.cnn}
peak        the best run of a family on a dataset (ties broken by lower training time);
            its cost is that run's training time
typical     the median score of the family's best-by-median classifier, robust to collapsed runs
time        wall-clock training time as logged (data preparation, embedding extraction and
            inference excluded)
device time time x accelerators held: 4 GPUs for every transformer run, 1 GPU for every
            static-embedding run (both confirmed in the run log), none for ML
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DERIVED = DATA / "derived"

FAMILIES = ["ML", "static", "transformer"]
DATASETS = ["bbc-news", "20newsgroups", "imdb", "arxiv_protoformer",
            "reuters21578", "ohsumed", "arxiv", "rcv1"]
SINGLE, MULTI = DATASETS[:4], DATASETS[4:]
MEASURES = ["macro", "micro"]
DEVICES = {"ML": 0, "static": 1, "transformer": 4}
Q05_K8 = 3.031  # Nemenyi q_0.05 for k = 8 (Demsar 2006, Table 5a)
TOLERANCES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
WCE_BASES = ["solo", "cat-doc", "lsa", "dot"]
ENCODERS, LARGE = ["bert", "roberta", "distilbert", "xlnet"], ["llama", "deepseek"]


def geomean(x):
    return float(np.exp(np.log(np.asarray(x, dtype=float)).mean()))


def best_rows(df, by):
    """Best run per group, ties broken by lower training time."""
    return df.sort_values(["value", "time_s"], ascending=[False, True]).groupby(by, as_index=False).first()


def verify_against_v6(runs):
    """Aggregate the runs exactly as the V6 tables did and count disagreements."""
    out = {}
    for measure in MEASURES:
        r = runs[runs.measure == measure]
        agg = r.groupby(["dataset", "classifier"]).agg(max=("value", "max"), mean=("value", "mean"),
                                                        median=("value", "median"), min=("value", "min"),
                                                        count=("value", "size"))
        dl = pd.read_csv(DATA / f"dl_by_dataset_{measure}.csv").set_index(["dataset", "classifier"])
        allc = pd.read_csv(DATA / "all_classifiers_by_dataset.csv")
        allc = allc[allc.measure == measure].set_index(["dataset", "classifier"])
        checks = mism = 0
        for pub, cols in [(dl, ["max", "mean", "median", "min"]), (allc, ["mean", "median"])]:
            j = pub.join(agg, rsuffix="_run")
            for c in cols:
                checks += len(j)
                mism += int(((j[c] - j[c + "_run"].round(3)).abs() > 0.0015).sum())
            checks += len(j)
            mism += int((j["count"] != j["count_run"]).sum())
        out[measure] = {"checks": checks, "mismatches": mism}
    return out


def peak_table(runs):
    b = best_rows(runs, ["measure", "dataset", "family"])
    b["devices"] = b.family.map(DEVICES)
    b["device_s"] = b.time_s * b.devices
    return b


def typical_table(runs):
    med = runs.groupby(["measure", "dataset", "family", "classifier"]).agg(
        median=("value", "median"), time_med=("time_s", "median")).reset_index()
    med = med.sort_values(["median", "time_med"], ascending=[False, True])
    return med.groupby(["measure", "dataset", "family"], as_index=False).first()


def paired_tests(peak):
    out = []
    for measure in MEASURES:
        p = peak[peak.measure == measure].pivot(index="dataset", columns="family", values="value").loc[DATASETS]
        for a, b in [("transformer", "ML"), ("transformer", "static"), ("ML", "static")]:
            d = (p[a] - p[b]).round(6)
            res = stats.wilcoxon(d[d != 0], method="exact")
            out.append(dict(measure=measure, a=a, b=b, mean_diff_pts=100 * d.mean(), median_diff_pts=100 * d.median(),
                            min_diff_pts=100 * d.min(), max_diff_pts=100 * d.max(),
                            rel_gain_pct=100 * d.mean() / p[b].mean(),
                            wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                            W=float(res.statistic), p=float(res.pvalue)))
    return pd.DataFrame(out)


def friedman(runs):
    out, ranks = {}, []
    fam = runs.drop_duplicates("classifier").set_index("classifier").family
    for measure in MEASURES:
        m = runs[runs.measure == measure].groupby(["dataset", "classifier"]).value.median().unstack().loc[DATASETS]
        r = m.rank(axis=1, ascending=False)
        chi2, p = stats.friedmanchisquare(*[m[c] for c in m.columns])
        k, n = m.shape[1], m.shape[0]
        out[measure] = dict(chi2=float(chi2), p=float(p), k=k, n=n, cd=float(Q05_K8 * np.sqrt(k * (k + 1) / (6 * n))))
        for clf, v in r.mean().sort_values().items():
            ranks.append(dict(measure=measure, classifier=clf, family=fam[clf], avg_rank=float(v)))
    return out, pd.DataFrame(ranks)


def cost_ratios(peak):
    rows = []
    for measure in MEASURES:
        p = peak[peak.measure == measure].set_index(["dataset", "family"])
        for ds in DATASETS:
            t, s, m = (p.loc[(ds, f)] for f in ["transformer", "static", "ML"])
            rows.append(dict(measure=measure, dataset=ds,
                             wall_T_over_ML=t.time_s / m.time_s, wall_T_over_S=t.time_s / s.time_s,
                             device_T_over_S=t.device_s / s.device_s,
                             gain_T_vs_ML_pts=100 * (t.value - m.value), gain_T_vs_S_pts=100 * (t.value - s.value)))
    df = pd.DataFrame(rows)
    summary = {}
    for measure in MEASURES:
        d = df[df.measure == measure]
        summary[measure] = {k: geomean(d[k]) for k in ["wall_T_over_ML", "wall_T_over_S", "device_T_over_S"]}
        summary[measure].update({f"range_{k}": [float(d[k].min()), float(d[k].max())]
                                 for k in ["wall_T_over_ML", "wall_T_over_S", "device_T_over_S"]})
    return df, summary


def by_task_type(ratios):
    out = {}
    for measure in MEASURES:
        g = ratios[ratios.measure == measure].set_index("dataset").gain_T_vs_ML_pts
        u = stats.mannwhitneyu(g.loc[MULTI], g.loc[SINGLE], alternative="two-sided", method="exact")
        out[measure] = dict(single_mean=float(g.loc[SINGLE].mean()), multi_mean=float(g.loc[MULTI].mean()),
                            single_range=[float(g.loc[SINGLE].min()), float(g.loc[SINGLE].max())],
                            multi_range=[float(g.loc[MULTI].min()), float(g.loc[MULTI].max())], p=float(u.pvalue))
    return out


def pareto_and_tolerance(runs):
    front, tol = [], []
    for (measure, ds), g in runs.groupby(["measure", "dataset"]):
        g = g.sort_values(["time_s", "value"], ascending=[True, False])
        best = -1
        for _, r in g.iterrows():
            if r.value > best:
                front.append(r)
                best = r.value
        top = best_rows(g, ["dataset"]).iloc[0]
        for eps in TOLERANCES:
            ok = g[g.value >= top.value - eps / 100 - 1e-9].sort_values("time_s").iloc[0]
            tol.append(dict(measure=measure, dataset=ds, eps_pts=eps, family=ok.family, classifier=ok.classifier,
                            lm=ok.lm, form=ok.form, config=ok.config, value=ok.value, time_s=ok.time_s,
                            best_value=top.value, best_classifier=top.classifier, best_lm=top.lm,
                            best_time_s=top.time_s, time_fraction=ok.time_s / top.time_s))
    return pd.DataFrame(front), pd.DataFrame(tol)


def repr_forms(runs):
    ml = runs[runs.family == "ML"]
    forms = ml.groupby(["measure", "dataset", "form"]).agg(max=("value", "max"), median=("value", "median"),
                                                           time_med=("time_s", "median"), n=("value", "size"))
    forms = forms.reset_index()
    solo = []
    for (measure, ds), g in forms.groupby(["measure", "dataset"]):
        b = g.sort_values(["max", "time_med"], ascending=[False, True]).iloc[0]
        s = g[g.form == "solo"].iloc[0]
        solo.append(dict(measure=measure, dataset=ds, best_form=b.form, best_max=b["max"], best_time=b.time_med,
                         solo_max=s["max"], solo_time=s.time_med, gap_pts=100 * (b["max"] - s["max"]),
                         time_ratio=b.time_med / s.time_med))
    return forms, pd.DataFrame(solo)


def paired(runs, keys, flag):
    """Best run with flag vs. best run without it, for every group defined by keys."""
    g = runs.groupby(keys + [flag]).value.max().unstack(flag).dropna()
    t = runs.groupby(keys + [flag]).time_s.median().unstack(flag).dropna()
    d = (100 * (g[True] - g[False])).rename("delta_pts").to_frame()
    d["time_ratio"] = (t[True] / t[False]).reindex(d.index)
    return d.reset_index()


def summarise_pairs(d):
    nz = d.delta_pts[d.delta_pts != 0]
    res = stats.wilcoxon(nz)
    return dict(n=len(d), mean=float(d.delta_pts.mean()), median=float(d.delta_pts.median()),
                up=int((d.delta_pts > 0).sum()), down=int((d.delta_pts < 0).sum()), ties=int((d.delta_pts == 0).sum()),
                p=float(res.pvalue), time_ratio_median=float(d.time_ratio.median()))


def class_embeddings(runs):
    """WCE in the static networks and in the classical module; TCE in the transformers."""
    out, frames = {}, []
    st = runs[runs.family == "static"]
    tr = runs[runs.family == "transformer"]
    ml = runs[runs.family == "ML"].copy()
    ml["base"] = ml.form.str.replace("-wce", "", regex=False)
    ml = ml[ml.base.isin(WCE_BASES)]
    tr_keys = ["dataset", "classifier", "lm", "vocab"]
    for measure in MEASURES:
        d1 = paired(st[st.measure == measure], ["dataset", "classifier", "lm", "tunable"], "wce").assign(kind="WCE, static NN")
        d2 = paired(tr[tr.measure == measure], tr_keys, "tce").assign(kind="TCE, transformer")
        d3 = paired(ml[ml.measure == measure], ["dataset", "classifier", "lm", "base"], "wce").assign(kind="WCE, classical ML")
        for d in (d1, d2, d3):
            d["measure"] = measure
            frames.append(d)
            out[f"{measure}|{d.kind.iloc[0]}"] = summarise_pairs(d)
        lowest = tr[tr.measure == measure].groupby(tr_keys).value.min().rename("min_run").reset_index()
        clean = d2.merge(lowest, on=tr_keys)
        out[f"{measure}|TCE, transformer, no collapsed runs"] = summarise_pairs(clean[clean.min_run >= 0.35])
        out[f"{measure}|WCE static by dataset"] = d1.groupby("dataset").delta_pts.mean().round(2).to_dict()
        out[f"{measure}|WCE ML by base"] = {b: summarise_pairs(d3[d3.base == b]) for b in WCE_BASES}
    return out, pd.concat(frames, ignore_index=True)


def language_models(runs):
    tr = runs[(runs.family == "transformer") & (runs.measure == "macro")]
    best = tr.groupby(["dataset", "lm"]).value.max().unstack().loc[DATASETS]
    tmed = tr.groupby(["dataset", "lm"]).time_s.median().unstack().loc[DATASETS]
    behind = 100 * (best.max(axis=1).values[:, None] - best)
    rank = best.rank(axis=1, ascending=False)
    rel_time = tmed.div(tmed["bert"], axis=0)
    table = pd.DataFrame({"avg_rank": rank.mean(), "mean_behind_pts": behind.mean(), "max_behind_pts": behind.max(),
                          "median_time_vs_bert": rel_time.median(), "wins": (behind == 0).sum()})
    large_gain = 100 * (best[LARGE].max(axis=1) - best[ENCODERS].max(axis=1))
    large_time = tmed[LARGE].median(axis=1) / tmed[ENCODERS].median(axis=1)
    fr = stats.friedmanchisquare(*[best[c] for c in best.columns])
    summary = dict(friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
                   large_vs_encoder_gain_pts=large_gain.round(2).to_dict(),
                   large_vs_encoder_gain_mean=float(large_gain.mean()),
                   large_vs_encoder_gain_median=float(large_gain.median()),
                   large_vs_encoder_time_geo=geomean(large_time),
                   large_vs_encoder_time_range=[float(large_time.min()), float(large_time.max())],
                   distil_vs_bert_time_median=float((tmed["distilbert"] / tmed["bert"]).median()),
                   distil_behind_best_encoder_pts=(100 * (best[ENCODERS].max(axis=1) - best["distilbert"])).round(2).to_dict())
    long = behind.stack().rename("behind_pts").reset_index().merge(
        tmed.stack().rename("time_med").reset_index(), on=["dataset", "lm"])
    return table.sort_values("avg_rank"), long, summary


def hardware(runs):
    h = runs[runs.os.notna()].copy()
    h["machine"] = np.where(h.family == "ML",
                            np.where(h.os == "macOS", "Apple M3 Max (CPU)", "Linux workstation (CPU)"),
                            h.gpu_name.fillna("unknown"))
    return h.groupby(["family", "dataset", "machine", "n_gpus"]).size().rename("runs").reset_index()


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(DATA / "runs.csv")

    verification = verify_against_v6(runs)
    peak = peak_table(runs)
    typical = typical_table(runs)
    tests = paired_tests(peak)
    fr, ranks = friedman(runs)
    ratios, ratio_summary = cost_ratios(peak)
    front, tol = pareto_and_tolerance(runs)
    forms, solo = repr_forms(runs)
    ce_summary, ce_pairs = class_embeddings(runs)
    lm_table, lm_long, lm_summary = language_models(runs)
    hw = hardware(runs)
    neural = runs[runs.family != "ML"]
    collapsed = neural[neural.value < 0.35]

    typ = {}
    for m in MEASURES:
        p = typical[typical.measure == m].pivot(index="dataset", columns="family", values="median")
        typ[m] = {"T_minus_S_pts": 100 * float((p.transformer - p.static).mean()),
                  "T_minus_ML_pts": 100 * float((p.transformer - p.ML).mean()),
                  "ML_minus_S_pts": 100 * float((p.ML - p.static).mean())}

    results = dict(verification=verification, friedman=fr, cost_ratios=ratio_summary, typical_gaps=typ,
                   by_task_type=by_task_type(ratios), class_embeddings=ce_summary, language_models=lm_summary,
                   peak_family_means={m: peak[peak.measure == m].groupby("family").value.mean().round(4).to_dict()
                                      for m in MEASURES},
                   pareto_counts={m: front[front.measure == m].family.value_counts().to_dict() for m in MEASURES},
                   run_counts=runs[runs.measure == "macro"].groupby("family").size().to_dict())
    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    for name, df in dict(peak=peak, typical=typical, wilcoxon=tests, ranks=ranks, cost_ratios=ratios,
                         pareto=front, tolerance=tol, repr_forms=forms, solo=solo, class_embedding_pairs=ce_pairs,
                         lm_table=lm_table.rename_axis("lm").reset_index(), lm_long=lm_long,
                         hardware=hw, collapsed=collapsed).items():
        df.to_csv(DERIVED / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    print(json.dumps({k: results[k] for k in ["verification", "cost_ratios", "typical_gaps", "by_task_type",
                                              "peak_family_means", "pareto_counts", "run_counts"]}, indent=1))
    print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
