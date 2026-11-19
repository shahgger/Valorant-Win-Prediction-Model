"""
Statistics for the NGES 2027 paper: logistic regression vs Extra Trees vs Random Forest, 15 training-data options
(weights by level and by age), and three input representations (engineered, rich, raw).

Reads the saved results of final_selection.py (figures/final_selection_cache/); trains nothing. Takes a few minutes.
    python paper_analysis.py
Writes figures/paper/*.csv (the appendix tables for GitHub) and prints a summary. Send the printed output and the
figures/paper folder back.

Comparisons on pro VCT 2026 (paired bootstrap by map, 2,000 resamples, 95% CI; Holm within each family):
  Q1 data:      each option vs the SAME algorithm and input set trained on B (pro 2023-2025); family = one
                (model, algorithm, input set): 14 comparisons
  Q2 algorithm: Extra Trees vs logistic regression, Random Forest vs logistic regression, Extra Trees vs Random
                Forest, on the same option and input set; family = one (model, input set, pair): 15 comparisons
  Q3 inputs:    rich vs engineered and raw vs engineered (pre-match: 11 match vs 4 final) for the same algorithm and
                option; family = one (model, algorithm, pair): 15 comparisons
Also: Brier score, expected calibration error, calibration bins, validation-test rank agreement, and the
log loss of each option against the share of its training weight that comes from 2021-2022.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

OUT = "figures/paper"
os.makedirs(OUT, exist_ok=True)
N_BOOT = 2000
ALGOS = ["Logistic Regression", "Extra Trees", "Random Forest"]
SHORT = {"Logistic Regression": "LR", "Extra Trees": "ET", "Random Forest": "RF"}
OPTS = ["B", "D", "D½", "C", "Cf", "E", "Eq½", "E½", "Eo½", "O", "OQ", "R¾", "R½", "RQ¾", "RQ½"]
REF_SET = {"prematch": "4 final", "live": "engineered", "round": "engineered"}
SETS = {"prematch": ["4 final", "11 match"], "live": ["engineered", "rich", "raw"],
        "round": ["engineered", "rich", "raw"]}


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_map(g, x):
    s = pd.Series(x).groupby(np.asarray(g)).agg(["sum", "size"]).sort_index()
    return s["sum"].to_numpy(), s["size"].to_numpy()


def compare(y, g, p_base, p_new, idx):
    """Mean log-loss gain of p_new over p_base (positive = new better), bootstrap CI and two-sided p."""
    b, cnt = per_map(g, ll(y, p_base))
    n, _ = per_map(g, ll(y, p_new))
    d = b - n
    boot = d[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pv = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()) + 1 / N_BOOT)
    return d.sum() / cnt.sum(), lo, hi, pv, "better" if lo > 0 else ("worse" if hi < 0 else "same")


def holm(df, key):
    df = df.copy()
    df["verdict_holm"] = "same"
    for _, grp in df.groupby(key):
        still = True
        for k, (i, r) in enumerate(grp.sort_values("p").iterrows()):
            still = still and r["p"] <= 0.05 / (len(grp) - k)
            if still and r["verdict"] != "same":
                df.loc[i, "verdict_holm"] = r["verdict"]
    return df


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())


def load(part):
    found, meta = {}, None
    for f in sorted(glob.glob(f"figures/final_selection_cache/{part}_*.pkl"), key=os.path.getmtime):
        if "_quick_" in os.path.basename(f):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        meta = meta or d["meta"]
        if len(d["meta"]["y_te"]) != len(meta["y_te"]):
            continue
        for c in d["finalists"]:
            if c["algorithm"] in ALGOS and c["option"] in OPTS:
                found[(c["option"], c["algorithm"], c["features"])] = c
    return found, meta


all_fin, all_data, all_alg, all_inp, all_cal, all_bins, summary = [], [], [], [], [], [], []
for part in ("prematch", "live", "round"):
    found, meta = load(part)
    if not found:
        print(f"{part}: nothing saved in figures/final_selection_cache/")
        continue
    y, g = meta["y_te"].astype(float), meta["g_te"]
    yv, gv = meta["y_val"].astype(float), meta["g_val"]
    k = len(np.unique(g))
    rng = np.random.default_rng(2027)
    idx = rng.integers(0, k, (N_BOOT, k))
    print(f"\n==================== {part}: test = pro 2026, {k:,} maps, {len(y):,} rows; "
          f"validation {len(np.unique(gv)):,} maps; finalists {len(found)}")
    # ---- every finalist
    for (o, a, f), c in found.items():
        p0 = c["test_preds"][0]
        lls = [ll(y, p).mean() for p in c["test_preds"].values()]
        all_fin.append({"model": part, "option": o, "algorithm": a, "input_set": f, "chosen_params": str(c["params"]),
                        "val_log_loss": c["val_log_loss"], "test_log_loss": lls[0], "test_ll_min_seeds": min(lls),
                        "test_ll_max_seeds": max(lls), "n_seeds": len(lls),
                        "test_accuracy": ((p0 > 0.5) == y).mean(), "test_brier": ((p0 - y) ** 2).mean(),
                        "test_ece": ece(y, p0)})
    # ---- Q1 data: option vs the same algorithm and input set on B
    rows = []
    for a in ALGOS:
        for f in SETS[part]:
            base = found.get(("B", a, f))
            if base is None:
                continue
            for o in OPTS[1:]:
                c = found.get((o, a, f))
                if c is None:
                    continue
                agree = []
                for s, p in c["test_preds"].items():
                    r = compare(y, g, base["test_preds"].get(s, base["test_preds"][0]), p, idx)
                    agree.append(r[4])
                    if s == 0:
                        gain, lo, hi, pv, v = r
                rows.append({"model": part, "algorithm": a, "input_set": f, "option": o, "vs": "B",
                             "gain": gain, "ci_low": lo, "ci_high": hi, "p": pv, "verdict": v,
                             "seeds_agree": f"{agree.count(v)} of {len(agree)}"})
    dq = holm(pd.DataFrame(rows), ["model", "algorithm", "input_set"])
    all_data.append(dq)
    # ---- Q2 algorithm: pairs on the same option and input set
    rows = []
    for f in SETS[part]:
        for a1, a2 in (("Logistic Regression", "Extra Trees"), ("Logistic Regression", "Random Forest"),
                       ("Random Forest", "Extra Trees")):
            for o in OPTS:
                b, c = found.get((o, a1, f)), found.get((o, a2, f))
                if b is None or c is None:
                    continue
                agree = []
                for s, p in c["test_preds"].items():
                    r = compare(y, g, b["test_preds"].get(s, b["test_preds"][0]), p, idx)
                    agree.append(r[4])
                    if s == 0:
                        gain, lo, hi, pv, v = r
                rows.append({"model": part, "input_set": f, "pair": f"{SHORT[a2]} vs {SHORT[a1]}", "option": o,
                             "gain": gain, "ci_low": lo, "ci_high": hi, "p": pv, "verdict": v,
                             "seeds_agree": f"{agree.count(v)} of {len(agree)}"})
    aq = holm(pd.DataFrame(rows), ["model", "input_set", "pair"])
    all_alg.append(aq)
    # ---- Q3 inputs: each other input set vs the reference input set
    rows = []
    ref = REF_SET[part]
    for a in ALGOS:
        for f in [s for s in SETS[part] if s != ref]:
            for o in OPTS:
                b, c = found.get((o, a, ref)), found.get((o, a, f))
                if b is None or c is None:
                    continue
                gain, lo, hi, pv, v = compare(y, g, b["test_preds"][0], c["test_preds"][0], idx)
                rows.append({"model": part, "algorithm": a, "pair": f"{f} vs {ref}", "option": o,
                             "gain": gain, "ci_low": lo, "ci_high": hi, "p": pv, "verdict": v})
    iq = holm(pd.DataFrame(rows), ["model", "algorithm", "pair"])
    all_inp.append(iq)
    # ---- calibration of each algorithm on B, its input set chosen on validation
    for a in ALGOS:
        cs = [c for (o, aa, f), c in found.items() if o == "B" and aa == a]
        if not cs:
            continue
        c = min(cs, key=lambda c: c["val_log_loss"])
        p = c["test_preds"][0]
        b = np.minimum((p * 10).astype(int), 9)
        for kb in range(10):
            sel = b == kb
            if sel.any():
                all_bins.append({"model": part, "algorithm": a, "input_set": c["features"],
                                 "bin": f"{kb / 10:.1f}-{(kb + 1) / 10:.1f}", "rows": int(sel.sum()),
                                 "mean_predicted": p[sel].mean(), "observed_win_rate": y[sel].mean()})
    # ---- validation-test agreement
    fin = pd.DataFrame([r for r in all_fin if r["model"] == part])
    rho = fin["val_log_loss"].rank().corr(fin["test_log_loss"].rank())
    # ---- printed summary
    print(f"  validation-test rank correlation (Spearman) over {len(fin)} finalists: {rho:.2f}")
    print("  best and worst option per algorithm and input set (test log loss, seed 0):")
    for (a, f), grp in fin.groupby(["algorithm", "input_set"]):
        b = grp.loc[grp["option"] == "B", "test_log_loss"]
        lo_r, hi_r = grp.loc[grp["test_log_loss"].idxmin()], grp.loc[grp["test_log_loss"].idxmax()]
        sub = dq[(dq["algorithm"] == a) & (dq["input_set"] == f)]
        cnt = lambda col: f"{(sub[col] == 'better').sum()}/{(sub[col] == 'same').sum()}/{(sub[col] == 'worse').sum()}"
        print(f"    {SHORT[a]} {f:11s} B {b.iloc[0] if len(b) else float('nan'):.4f} | best {lo_r['option']:4s} "
              f"{lo_r['test_log_loss']:.4f} | worst {hi_r['option']:4s} {hi_r['test_log_loss']:.4f} | "
              f"spread {hi_r['test_log_loss'] - lo_r['test_log_loss']:.4f} | data verdicts b/s/w before "
              f"{cnt('verdict')} after Holm {cnt('verdict_holm')}")
        summary.append({"model": part, "algorithm": a, "input_set": f, "B_test_log_loss": b.iloc[0] if len(b) else None,
                        "best_option": lo_r["option"], "best_test_log_loss": lo_r["test_log_loss"],
                        "worst_option": hi_r["option"], "worst_test_log_loss": hi_r["test_log_loss"],
                        "spread": hi_r["test_log_loss"] - lo_r["test_log_loss"],
                        "data_better_before": (sub["verdict"] == "better").sum(),
                        "data_same_before": (sub["verdict"] == "same").sum(),
                        "data_worse_before": (sub["verdict"] == "worse").sum(),
                        "data_better_holm": (sub["verdict_holm"] == "better").sum(),
                        "data_same_holm": (sub["verdict_holm"] == "same").sum(),
                        "data_worse_holm": (sub["verdict_holm"] == "worse").sum(),
                        "val_test_spearman": rho})
    print("  data verdicts that hold after Holm:")
    held = dq[dq["verdict_holm"] != "same"]
    print(held[["algorithm", "input_set", "option", "gain", "ci_low", "ci_high", "verdict_holm",
                "seeds_agree"]].round(4).to_string(index=False) if len(held) else "    none")
    print("  algorithm verdicts (counts better/same/worse of the second-named algorithm), before -> after Holm:")
    for (f, pr), grp in aq.groupby(["input_set", "pair"]):
        c1 = [int((grp["verdict"] == v).sum()) for v in ("better", "same", "worse")]
        c2 = [int((grp["verdict_holm"] == v).sum()) for v in ("better", "same", "worse")]
        mg = grp["gain"].mean()
        print(f"    {f:11s} {pr:9s} mean gain {mg:+.4f}  before {c1}  after Holm {c2}")
    print("  input-set verdicts, before -> after Holm:")
    for (a, pr), grp in iq.groupby(["algorithm", "pair"]):
        c1 = [int((grp["verdict"] == v).sum()) for v in ("better", "same", "worse")]
        c2 = [int((grp["verdict_holm"] == v).sum()) for v in ("better", "same", "worse")]
        print(f"    {SHORT[a]} {pr:24s} mean gain {grp['gain'].mean():+.4f}  before {c1}  after Holm {c2}")
    print("  calibration on B (input set chosen on validation): Brier, ECE")
    for a in ALGOS:
        r = fin[(fin["algorithm"] == a) & (fin["option"] == "B")]
        if len(r):
            r = r.loc[r["val_log_loss"].idxmin()]
            print(f"    {SHORT[a]} {r['input_set']:11s} log loss {r['test_log_loss']:.4f}  acc {r['test_accuracy']:.4f}  "
                  f"Brier {r['test_brier']:.4f}  ECE {r['test_ece']:.4f}  params {r['chosen_params']}")

pd.DataFrame(all_fin).sort_values(["model", "algorithm", "input_set", "option"]).to_csv(f"{OUT}/finalists.csv", index=False)
pd.concat(all_data).to_csv(f"{OUT}/q1_data_options.csv", index=False)
pd.concat(all_alg).to_csv(f"{OUT}/q2_algorithms.csv", index=False)
pd.concat(all_inp).to_csv(f"{OUT}/q3_input_sets.csv", index=False)
pd.DataFrame(all_bins).to_csv(f"{OUT}/calibration_bins.csv", index=False)
pd.DataFrame(summary).to_csv(f"{OUT}/summary_by_algorithm_input_set.csv", index=False)
print(f"\nSaved {OUT}/: finalists.csv, q1_data_options.csv, q2_algorithms.csv, q3_input_sets.csv, "
      f"calibration_bins.csv, summary_by_algorithm_input_set.csv. Send this output and the folder back.")
