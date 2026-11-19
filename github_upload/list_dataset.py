"""
Lists everything in the data folder: every folder, every file (CSV "sheet"), its number of rows and columns, and for
every column its type, how many values are missing, how many different values it has and an example value.

Run from the valorant-fyp folder:   python list_dataset.py
Writes figures/dataset_files.csv (one row per file), figures/dataset_columns.csv (one row per column) and
       figures/dataset_structure.txt (the same as printed, to send back).
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(errors="replace")       # odd characters in team names do not crash Windows consoles
ROOT = "data"
os.makedirs("figures", exist_ok=True)
files, cols, lines = [], [], []


def say(text=""):
    print(text)
    lines.append(text)


say(f"Folder tree of '{ROOT}/'")
for folder, subdirs, names in sorted(os.walk(ROOT)):
    subdirs.sort()
    depth = folder.count(os.sep)
    say("  " * depth + os.path.basename(folder) + "/")
    for n in sorted(names):
        size = os.path.getsize(os.path.join(folder, n)) / 1e6
        say("  " * (depth + 1) + f"{n}  ({size:.1f} MB)")

say("\nEvery CSV file: rows, columns and each column")
for folder, _, names in sorted(os.walk(ROOT)):
    for n in sorted(names):
        if not n.lower().endswith(".csv"):
            continue
        path = os.path.join(folder, n)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            say(f"\n{path}: could not read ({e})")
            continue
        files.append({"file": path, "rows": len(df), "columns": df.shape[1]})
        say(f"\n{path}: {len(df):,} rows x {df.shape[1]} columns")
        for c in df.columns:
            s = df[c]
            miss = s.isna().mean()
            example = s.dropna().iloc[0] if s.notna().any() else ""
            example = str(example)[:40]
            cols.append({"file": path, "column": c, "type": str(s.dtype), "missing_pct": round(miss * 100, 1),
                         "distinct_values": s.nunique(), "example": example})
            say(f"   {str(c):35s} {str(s.dtype):8s} missing {miss:6.1%}  distinct {s.nunique():>8,}  e.g. {example}")

pd.DataFrame(files).to_csv("figures/dataset_files.csv", index=False)
pd.DataFrame(cols).to_csv("figures/dataset_columns.csv", index=False)
with open("figures/dataset_structure.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
say(f"\n{len(files)} CSV files, {len(cols)} columns in total.")
say("Saved figures/dataset_files.csv, figures/dataset_columns.csv and figures/dataset_structure.txt "
    "(send the .txt file back).")
