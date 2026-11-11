"""Build the run-level table used by the revised analysis.

Sources (copied from out/20250411.lc_full_dataset_04102025.tsv/analysis_data/, the
final results-analysis output behind the published paper):
  data/runs/<dataset>_results_2025-04-11.csv   one row per run and metric; complete, but
                                                 training time rounded to whole seconds
  data/runs/layercake.all.summary.2025-04-11.csv the same runs with full-precision training
                                                 time, OS, CPU and GPU, but deduplicated,
                                                 so it lacks some runs

The per-dataset files are the source of record: aggregated, they reproduce every
classifier-by-dataset statistic published in the earlier paper (see analyze.py,
verify_against_v6). Where a run can be matched unambiguously in the full log, its
precise time and hardware are attached.

Usage: python extract_runs.py  (run from the analysis/ directory)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs"
OUT = HERE.parent / "data" / "runs.csv"

FAMILY = {"lr": "ML", "svm": "ML", "nb": "ML", "cnn": "static", "lstm": "static", "attn": "static",
          "hf.sc": "transformer", "hf.cnn": "transformer"}


def main():
    files = sorted(RUNS.glob("*_results_2025-04-11.csv"))
    r = pd.concat([pd.read_csv(f, thousands=",") for f in files], ignore_index=True)
    r.columns = ["dataset", "classifier", "class_type", "comp_method", "lm", "form", "representation",
                 "dims", "measure", "value", "time_logged"]
    r = r[r.measure.isin(["final-te-macro-f1", "final-te-micro-f1"])].copy()
    r["measure"] = r.measure.str.extract(r"(macro|micro)", expand=False)
    r["family"] = r.classifier.map(FAMILY)

    full = pd.read_csv(RUNS / "layercake.all.summary.2025-04-11.csv")
    full = full[full.measure.isin(["final-te-macro-f1", "final-te-micro-f1"])].copy()
    full["measure"] = full.measure.str.extract(r"(macro|micro)", expand=False)
    full["lm"] = full.embeddings.str.split(":").str[0]
    full["os"] = full.os.str.extract(r"^(\w+)", expand=False)
    full["gpu_name"] = full.gpus.astype(str).str.extract(r"'name': '([^']+)'", expand=False)
    full["n_gpus"] = pd.to_numeric(full.gpus.astype(str).str.extract(r"^(\d+):", expand=False))

    key = ["dataset", "classifier", "measure", "lm", "dimk", "v3"]
    r["dimk"] = r.dims.str.replace(" ", "")
    full["dimk"] = full.dimensions.str.replace(" ", "")
    r["v3"] = r.value.round(3)
    full["v3"] = full.value.round(3)
    # keep only keys that identify a single run in the full log
    g = full.groupby(key)
    uniq = g.filter(lambda x: len(x) == 1)[key + ["timelapse", "os", "cpus", "gpu_name", "n_gpus"]]
    r = r.merge(uniq, on=key, how="left")
    ok = r.timelapse.notna() & ((r.timelapse - r.time_logged).abs() <= 1.0)
    r["time_s"] = np.where(ok, r.timelapse, r.time_logged).astype(float)
    r["time_precise"] = ok
    for c in ["os", "cpus", "gpu_name", "n_gpus"]:
        r.loc[~ok, c] = np.nan

    r["tce"] = r.representation.str.contains("tce", na=False)
    r["wce"] = np.where(r.family == "ML", r.form.str.endswith("-wce"),
                        r.representation.str.contains("wce", na=False))
    r["tunable"] = r.representation.str.contains(r"\.tunable", na=False)
    r["vocab"] = pd.to_numeric(r.dims.str.extract(r"^\((\d+),", expand=False), errors="coerce")
    r["config"] = r.representation.str.extract(r"->(.*):\[def\]$", expand=False)

    cols = ["dataset", "class_type", "family", "classifier", "lm", "form", "config", "representation", "dims",
            "vocab", "tce", "wce", "tunable", "measure", "value", "time_s", "time_logged", "time_precise",
            "os", "cpus", "gpu_name", "n_gpus"]
    r[cols].to_csv(OUT, index=False)
    print(f"{len(r)} rows ({r.time_precise.mean():.1%} with precise time, "
          f"{r.os.notna().mean():.1%} with hardware) -> {OUT.name}")


if __name__ == "__main__":
    main()
