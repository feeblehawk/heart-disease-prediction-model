"""
Dataset completeness checker for the UCI Heart Disease files.
  processed.cleveland.data, processed.hungarian.data,
  processed.switzerland.data, processed.va.data
"""
import os
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
FEATURES = ["cp", "thalach", "oldpeak", "exang"]          # my 4 selected inputs
FILES = {
    "Cleveland":   "processed.cleveland.data",
    "Hungarian":   "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
ALLOWED = {"sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
           "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3}, "thal": {3, 6, 7},
           "num": {0, 1, 2, 3, 4}}
RANGES = {"age": (20, 100), "trestbps": (60, 250), "chol": (100, 600),
          "thalach": (60, 220), "oldpeak": (0, 8)}


def check(name, path):
    print("=" * 70)
    print(f"{name}  ({path})")
    print("=" * 70)
    if not os.path.exists(path):
        print("  FILE NOT FOUND - skipped\n")
        return None

    # '?' marks missing values in these files, so convert it to NaN
    df = pd.read_csv(path, names=COLS, na_values="?")

    # 1) size
    print(f"1) Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2) missing values
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(1)
    print(f"2) Total missing cells: {int(miss.sum())}")
    report = pd.DataFrame({"missing": miss, "percent": pct})
    print(report[report["missing"] > 0].to_string() if miss.sum() else "   no missing values")
    mostly_empty = list(pct[pct > 50].index)
    if mostly_empty:
        print(f"   Columns more than 50% empty (unusable): {mostly_empty}")

    # 3) data types
    bad = [c for c in COLS if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"3) Non-numeric columns: {bad if bad else 'none (all numeric)'}")

    # 4) duplicates
    print(f"4) Duplicate rows: {int(df.duplicated().sum())}")

    # 5) allowed values for categorical columns
    problems = []
    for c, ok in ALLOWED.items():
        found = set(df[c].dropna().unique())
        if not found <= ok:
            problems.append(f"{c}: unexpected {sorted(found - ok)}")
    print(f"5) Invalid category values: {problems if problems else 'none'}")

    # 6) numeric ranges + hidden missing values stored as 0
    out = []
    for c, (lo, hi) in RANGES.items():
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n:
            out.append(f"{c}: {n} outside {lo}-{hi}")
    print(f"6) Out-of-range values: {out if out else 'none'}")
    zeros_chol = int((df["chol"] == 0).sum())
    zeros_bp = int((df["trestbps"] == 0).sum())
    if zeros_chol or zeros_bp:
        print(f"   WARNING hidden missing: chol==0 -> {zeros_chol} rows, trestbps==0 -> {zeros_bp} rows")

    # 7) class balance
    y = (df["num"] > 0).astype(int)
    print(f"7) Target balance: healthy={int((y == 0).sum())} ({(y == 0).mean():.0%}), "
          f"disease={int((y == 1).sum())} ({(y == 1).mean():.0%})")

    # 8) my 4 selected features
    feat_miss = df[FEATURES].isnull().sum()
    feat_rows = int(df[FEATURES].isnull().any(axis=1).sum())
    print(f"8) Selected features {FEATURES}")
    print(f"   missing per feature: {feat_miss.to_dict()}  -> rows lost if dropped: {feat_rows}")

    # verdict
    if miss.sum() == 0:
        verdict = "COMPLETE"
    elif feat_rows == 0:
        verdict = "USABLE (gaps only in columns I do not use)"
    elif feat_rows / len(df) < 0.10:
        verdict = "USABLE after dropping a few rows"
    else:
        verdict = "NOT RECOMMENDED (too many gaps in my 4 features)"
    print(f"\n   VERDICT: {verdict}\n")
    return {"Dataset": name, "Rows": len(df), "Missing cells": int(miss.sum()),
            "Rows lost (4 features)": feat_rows, "Disease %": round(y.mean() * 100),
            "Verdict": verdict}


if __name__ == "__main__":
    rows = [r for r in (check(n, p) for n, p in FILES.items()) if r]
    if rows:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(rows).to_string(index=False))