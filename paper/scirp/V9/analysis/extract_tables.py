"""Extract the results tables from the published (V6) manuscript into CSV files.

The original run logs are not available locally, so the summary tables printed in
the V6 manuscript are the source of record for the revision's re-analysis. Every
number used downstream comes from one of these CSVs, so the provenance of each
figure and statistic can be traced back to a V6 table label.

Usage: python extract_tables.py  (run from the analysis/ directory)
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V6_TEX = HERE.parent.parent / "V6" / "Layer Cake 07312025.tex"
OUT = HERE.parent / "data"


def table_body(tex, label):
    """Return the rows between \\midrule/\\toprule and \\bottomrule for a labelled table."""
    i = tex.index("\\label{" + label + "}")
    start = tex.rfind("\\begin{table}", 0, i)
    end = tex.index("\\end{table}", i)
    block = tex[start:end]
    body = block.split("\\toprule", 1)[1].split("\\bottomrule", 1)[0]
    rows = []
    for line in body.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if not line or line.startswith("\\textbf{Dataset}") or line.startswith("\\textbf{Clf}") \
                or line.startswith("\\textbf{Classifier}") or line.startswith("& \\textbf"):
            continue
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).replace("\\_", "_").strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def carry_first(rows):
    """Fill blank first cells (dataset written once per block) downwards."""
    last = None
    for r in rows:
        if r[0]:
            last = r[0]
        else:
            r[0] = last
    return rows


def num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c])
    return df


def main():
    tex = V6_TEX.read_text()
    OUT.mkdir(exist_ok=True)

    # DL classifiers by dataset (Max/Mean/Median/Min/Std/Count/Time), V6 Tables 9-10
    for measure, label in [("macro", "tab:dl_classifier_macro_f1"), ("micro", "tab:dl_classifier_micro_f1")]:
        rows = carry_first(table_body(tex, label))
        df = pd.DataFrame(rows, columns=["dataset", "classifier", "max", "mean", "median", "min", "std", "count", "time_s"])
        df = num(df, ["max", "mean", "median", "min", "std", "count", "time_s"])
        df.insert(2, "measure", measure)
        df.to_csv(OUT / f"dl_by_dataset_{measure}.csv", index=False)

    # All 8 classifiers by dataset (Mean/Median/Std/Count/Time), V6 appendix Tables 14-17
    frames = []
    for measure, labels in [("macro", ["tab:global_classifier_macro_f1_single_label_summary",
                                       "tab:global_classifier_macro_f1_multi_label_summary"]),
                            ("micro", ["tab:global_classifier_micro_f1_single_label_summary",
                                       "tab:global_classifier_micro_f1_multi_label_summary"])]:
        for label in labels:
            rows = carry_first(table_body(tex, label))
            df = pd.DataFrame(rows, columns=["dataset", "classifier", "mean", "median", "std", "count", "time_s"])
            df = num(df, ["mean", "median", "std", "count", "time_s"])
            df.insert(2, "measure", measure)
            frames.append(df)
    pd.concat(frames).to_csv(OUT / "all_classifiers_by_dataset.csv", index=False)

    # ML top-5 configurations per dataset, V6 Tables 4-5
    for measure, label in [("macro", "tab:ml_classifier_summary_macro_f1"), ("micro", "tab:ml_classifier_summary_micro_f1")]:
        rows = table_body(tex, label)
        df = pd.DataFrame(rows, columns=["classifier", "dataset", "dims", "representation", "value", "time_s"])
        df = num(df, ["value", "time_s"])
        df["representation"] = df["representation"].str.replace(r":def$", "", regex=True)
        df.insert(2, "measure", measure)
        df["rank"] = df.groupby("dataset").cumcount() + 1
        df.to_csv(OUT / f"ml_top5_{measure}.csv", index=False)

    # ML representation forms per dataset, V6 Tables 6-7
    for measure, label in [("macro", "tab:macro_f1_repr_form"), ("micro", "tab:micro_f1_repr_form")]:
        rows = carry_first(table_body(tex, label))
        df = pd.DataFrame(rows, columns=["dataset", "representation_form", "mean", "max", "std", "count", "time_s"])
        df = num(df, ["mean", "max", "std", "count", "time_s"])
        df.insert(2, "measure", measure)
        df.to_csv(OUT / f"ml_repr_forms_{measure}.csv", index=False)

    # Global classifier summary, V6 Table 11
    rows = table_body(tex, "tab:avg_macro_micro_f1")
    df = pd.DataFrame(rows, columns=["classifier", "measure", "mean", "median", "std", "count", "time_s"])
    df = num(df, ["mean", "median", "std", "count", "time_s"])
    df["measure"] = df["measure"].str.extract(r"(macro|micro)")
    df.to_csv(OUT / "global_by_classifier.csv", index=False)

    # Dataset splits by classifier type, V6 Table 3
    rows = table_body(tex, "tab:dataset-summary")
    rows = [r for r in rows if len(r) == 9 and r[0] != "Classifier Type"]
    rows = carry_first(rows)
    df = pd.DataFrame(rows, columns=["classifier_type", "dataset", "type", "classes", "docs", "train", "val", "test", "train_pct"])
    for c in ["classes", "docs", "train", "test"]:
        df[c] = pd.to_numeric(df[c].str.replace(",", ""))
    df["val"] = pd.to_numeric(df["val"].str.replace(",", "").replace("NA", ""), errors="coerce")
    df.to_csv(OUT / "dataset_splits.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name:40s} {len(pd.read_csv(f)):4d} rows")


if __name__ == "__main__":
    main()
