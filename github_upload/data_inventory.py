"""
What data the project actually has, and why it is too little (and too noisy) for complex models to find hidden
patterns that logistic regression misses.

  1. Inventory: maps, teams, maps with round data and round states, per season and level.
  2. Rows are not independent results: the live model has ~20 rows per map, but every round of a map shares ONE
     answer (who won the map), so the real sample size is the number of maps.
  3. Thin coverage: how many different compositions and live game states exist, and how many of them appear
     often enough (30+ times) to learn anything from.
  4. Luck: how often the favourite still loses, and how a model that can learn ANY pattern (a lookup table of
     the win rate of every exact game state) does against logistic regression.
  5. Hidden-pattern test: logistic regression with every pairwise interaction of its inputs (the kind of pattern
     trees look for) against plain logistic regression, on the same test maps (paired bootstrap by map).
  6. Test-set precision: how big a difference the test set can detect at all.

Same splits as the main results: pro 2023-2025 for training, pro 2026 for testing.
Run from the valorant-fyp folder:   python data_inventory.py      (a few minutes)
Writes figures/data_inventory_*.csv
"""
import math
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from round_model import MAP_FEATURES, SEMI_TIERS, build_rounds, match_tier, state_features
from valorant_utils import ROLES

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
pd.set_option("display.width", 200)
MIN_SEEN = 30
FINAL = ["elo_tuned", "player_diff", "fam_diff", "map_pick"]
rng = np.random.default_rng(0)

data = pd.read_csv("data/maps_dataset_features.csv")


def level(row):
    if row["year"] >= 2023:
        return "pro"
    return "qualifier" if match_tier(row["Tournament"], row["Stage"]) in SEMI_TIERS else "pro-level 2021-22"


data["level"] = data.apply(level, axis=1)
years = sorted(data["year"].unique())
have = [y for y in years if os.path.exists(f"data/vct_{y}/matches/eco_rounds.csv")]
print("Building round tables (one per season)...")
rounds = pd.concat([build_rounds(data, [y], verbose=False) for y in have], ignore_index=True)
rounds = rounds.merge(data[["Game ID", "level"]].drop_duplicates("Game ID"), on="Game ID", how="left")

# ------------------------------------------------------------------ 1. inventory
print("\n1. What we have")
inv = []
for (yr, lv), g in data.groupby(["year", "level"]):
    r = rounds[(rounds["year"] == yr) & (rounds["level"] == lv)]
    teams = pd.concat([g["Team A"], g["Team B"]])
    per_team = teams.value_counts()
    inv.append({"season": yr, "level": lv, "maps": len(g), "teams": teams.nunique(),
                "median maps per team": int(per_team.median()),
                "maps with round data": r["Game ID"].nunique(), "round states": len(r)})
inv = pd.DataFrame(inv)
print(inv.to_string(index=False))
inv.to_csv("figures/data_inventory_overview.csv", index=False)
pro = data[data["level"] == "pro"]
print(f"   Pro 2023-2026 in total: {len(pro):,} maps. For comparison, image or text models are usually trained on "
      f"hundreds of thousands to millions of examples.")

# ------------------------------------------------------------------ 2. rows vs independent results
print("\n2. Rows are not independent results (live model, pro 2023-2025 training data)")
tr_r = rounds[(rounds["level"] == "pro") & rounds["year"].between(2023, 2025)]
te_r = rounds[(rounds["level"] == "pro") & (rounds["year"] == 2026)]
n_rows, n_maps = len(tr_r), tr_r["Game ID"].nunique()
print(f"   {n_rows:,} round states, but only {n_maps:,} different map results "
      f"({n_rows / n_maps:.1f} rows share each answer).")
print(f"   For the map winner, a model learns from {n_maps:,} outcomes, not {n_rows:,}. Trees split the data into "
      f"smaller and smaller groups; with ~{n_maps:,} outcomes the groups quickly become too small to trust.")

# ------------------------------------------------------------------ 3. coverage
print("\n3. How thinly the possibilities are covered")
n_agents = len(ROLES)
possible = math.comb(n_agents, 5)
comps = Counter(c for col in ["comp_A", "comp_B"] for c in pro[col].astype(str))
seen_often = sum(1 for v in comps.values() if v >= MIN_SEEN)
cells = Counter((m, c) for m, col_a, col_b in zip(pro["Map"], pro["comp_A"], pro["comp_B"]) for c in (col_a, col_b))
cells_often = sum(1 for v in cells.values() if v >= MIN_SEEN)
cov = [
    {"what": "possible 5-agent compositions", "count": possible},
    {"what": "different compositions actually played (pro 2023-2026)", "count": len(comps)},
    {"what": f"compositions played {MIN_SEEN}+ times", "count": seen_often},
    {"what": "different (map, composition) pairs played", "count": len(cells)},
    {"what": f"(map, composition) pairs played {MIN_SEEN}+ times", "count": cells_often},
]


def state_key(df, elo_edges):
    return list(zip(df["sA"].clip(upper=12), df["sB"].clip(upper=12), df["a_attack"], df["buy_a"].astype(str),
                    df["buy_b"].astype(str), np.digitize(df["elo_tuned"], elo_edges)))


elo_edges = np.quantile(tr_r["elo_tuned"], [0.2, 0.4, 0.6, 0.8])
tr_keys, te_keys = state_key(tr_r, elo_edges), state_key(te_r, elo_edges)
st_count = Counter(tr_keys)
possible_states = 13 * 13 * 2 * 16 * 5
te_thin = np.mean([st_count.get(k, 0) < MIN_SEEN for k in te_keys])
cov += [
    {"what": "possible live game states (score A x score B x side x both buys x 5 Elo bands)", "count": possible_states},
    {"what": "live game states seen in pro 2023-2025 training", "count": len(st_count)},
    {"what": f"live game states seen {MIN_SEEN}+ times", "count": sum(1 for v in st_count.values() if v >= MIN_SEEN)},
    {"what": f"% of 2026 test rows in a state seen under {MIN_SEEN} times in training", "count": round(te_thin * 100, 1)},
]
cov = pd.DataFrame(cov)
cov["count"] = [f"{int(v):,}" if float(v).is_integer() else f"{v}" for v in cov["count"]]
print(cov.to_string(index=False))
cov.to_csv("figures/data_inventory_coverage.csv", index=False)
print(f"   Of {len(comps):,} compositions played, only {seen_often:,} appear {MIN_SEEN}+ times, and on a given map "
      f"only {cells_often:,} do. A pattern like 'this composition on this map beats that one' has a handful of "
      f"games behind it at most.")


# ------------------------------------------------------------------ helpers
def boot_gain(loss_base, loss_new, groups, n=2000):
    """Paired bootstrap by map: mean log-loss gain of the new model (positive = new model better)."""
    g = pd.factorize(groups)[0]
    sb, sn, cnt = np.bincount(g, loss_base), np.bincount(g, loss_new), np.bincount(g)
    k = len(cnt)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        gains.append((sb[idx].sum() - sn[idx].sum()) / cnt[idx].sum())
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return (sb.sum() - sn.sum()) / cnt.sum(), lo, hi


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_lr(Xtr, ytr, groups, poly=False):
    """C chosen on the last 20% of the training maps (in time order), then refitted on all training rows."""
    order = pd.unique(groups)
    val = np.isin(groups, order[int(len(order) * 0.8):])
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        steps = [StandardScaler()] + ([PolynomialFeatures(2, include_bias=False), StandardScaler()] if poly else [])
        m = make_pipeline(*steps, LogisticRegression(C=C, max_iter=3000)).fit(Xtr[~val], ytr[~val])
        s = log_loss(ytr[val], m.predict_proba(Xtr[val])[:, 1])
        if best is None or s < best[0]:
            best = (s, C)
    steps = [StandardScaler()] + ([PolynomialFeatures(2, include_bias=False), StandardScaler()] if poly else [])
    return make_pipeline(*steps, LogisticRegression(C=best[1], max_iter=3000)).fit(Xtr, ytr), best[1]


# ------------------------------------------------------------------ 4. noise ceiling
print("\n4. How much is luck: the noise ceiling")
pm_cols = FINAL + ["A_won", "Game ID", "year"]
pm = pro[pm_cols].dropna().sort_values("Game ID")
ptr, pte = pm[pm["year"] <= 2025], pm[pm["year"] == 2026]
pm_lr, pm_C = fit_lr(ptr[FINAL].to_numpy(), ptr["A_won"].to_numpy(), ptr["Game ID"].to_numpy())
p_pm = pm_lr.predict_proba(pte[FINAL].to_numpy())[:, 1]
fav = np.maximum(p_pm, 1 - p_pm)
fav_won = np.where(p_pm >= 0.5, pte["A_won"], 1 - pte["A_won"])
bands = pd.cut(fav, [0.5, 0.55, 0.6, 0.65, 0.7, 1.0], include_lowest=True,
               labels=["50-55%", "55-60%", "60-65%", "65-70%", "70%+"])
noise = (pd.DataFrame({"band": bands, "fav": fav, "won": fav_won}).groupby("band", observed=True)
         .agg(maps=("won", "size"), predicted=("fav", "mean"), favourite_won=("won", "mean")))
noise["upsets"] = 1 - noise["favourite_won"]
print("   Pre-match model (pro 2023-2025 -> 2026): how often the favourite wins, by how sure the model is")
print((noise * [1, 100, 100, 100]).round(1).to_string())
noise.to_csv("figures/data_inventory_upsets.csv")
print(f"   {(fav < 0.6).mean():.0%} of 2026 maps are close to a coin flip before the map (favourite under 60%).")

ytr_l, yte_l = tr_r["A_won"].to_numpy(), te_r["A_won"].to_numpy()
Xtr_l = state_features(tr_r)[MAP_FEATURES].to_numpy()
Xte_l = state_features(te_r)[MAP_FEATURES].to_numpy()
live_lr, live_C = fit_lr(Xtr_l, ytr_l, tr_r["Game ID"].to_numpy())
p_live = live_lr.predict_proba(Xte_l)[:, 1]


def lookup(keys_fit, y_fit, keys_use, k=10):
    """Win rate of every exact game state, smoothed toward 50% (k pseudo-games)."""
    wins, n = Counter(), Counter(keys_fit)
    for key, yy in zip(keys_fit, y_fit):
        wins[key] += yy
    return np.array([(wins[key] + 0.5 * k) / (n[key] + k) for key in keys_use])


p_table = lookup(tr_keys, ytr_l, te_keys)                 # learned from training: the most flexible honest model
ceil = pd.DataFrame([
    {"model": "Logistic regression (engineered features)", "log_loss": log_loss(yte_l, p_live),
     "accuracy": ((p_live > 0.5) == yte_l).mean()},
    {"model": "Lookup table: win rate of each exact game state, from training", "log_loss": log_loss(yte_l, p_table),
     "accuracy": ((p_table > 0.5) == yte_l).mean()},
])
print("\n   Live model on pro 2026: logistic regression against the most flexible possible models")
print(ceil.round(4).to_string(index=False))
ceil.to_csv("figures/data_inventory_ceiling.csv", index=False)
print("   The lookup table can learn ANY pattern between game state and result (no straight-line assumption), but\n"
      "   each state has too few games behind it, so it does worse than logistic regression, which shares information\n"
      "   across states. This is what 'not enough data for hidden patterns' looks like.")

# ------------------------------------------------------------------ 5. hidden-pattern test
print("\n5. Hidden-pattern test: plain logistic regression vs every pairwise interaction of its inputs")
res = []
for name, (Xa, ya, ga, Xb, yb, gb) in {
    "Pre-match (4 final features)": (ptr[FINAL].to_numpy(), ptr["A_won"].to_numpy(), ptr["Game ID"].to_numpy(),
                                     pte[FINAL].to_numpy(), pte["A_won"].to_numpy(), pte["Game ID"].to_numpy()),
    "Live, map winner (10 engineered features)": (Xtr_l, ytr_l, tr_r["Game ID"].to_numpy(),
                                                  Xte_l, yte_l, te_r["Game ID"].to_numpy()),
}.items():
    base, cb = fit_lr(Xa, ya, ga)
    inter, ci = fit_lr(Xa, ya, ga, poly=True)
    pb, pi = base.predict_proba(Xb)[:, 1], inter.predict_proba(Xb)[:, 1]
    gain, lo, hi = boot_gain(row_loss(yb, pb), row_loss(yb, pi), gb)
    verdict = "better" if lo > 0 else ("worse" if hi < 0 else "same")
    n_in = Xa.shape[1]
    res.append({"model": name, "inputs": n_in, "inputs with interactions": n_in + n_in * (n_in + 1) // 2,
                "plain LR log loss": log_loss(yb, pb), "with interactions": log_loss(yb, pi),
                "gain": gain, "ci_low": lo, "ci_high": hi, "verdict": verdict})
res = pd.DataFrame(res)
print(res.round(4).to_string(index=False))
res.to_csv("figures/data_inventory_interactions.csv", index=False)
print("   Gain > 0 means the interactions helped. 'same' means the extra combinations found nothing reliable.")

# ------------------------------------------------------------------ 6. test-set precision
print("\n6. How small a difference the 2026 test set can detect")
n_pm, n_live = len(pte), te_r["Game ID"].nunique()
for name, n in [("pre-match", n_pm), ("live", n_live)]:
    print(f"   {name}: {n:,} test maps -> accuracy is known only to about +/-{1.96 * math.sqrt(0.25 / n) * 100:.1f} "
          f"points (95%)")
for r in res.itertuples():
    print(f"   {r.model}: the 95% interval of a log-loss difference is about {(r.ci_high - r.ci_low) / 2:.4f} wide "
          f"either side")
print("   Algorithm differences in the live and round results were 0.0005-0.002 in log loss, about the same size as "
      "this noise,\n   which is why most algorithms come out 'same'.")
print("\nSaved figures/data_inventory_*.csv. Send this output back.")
