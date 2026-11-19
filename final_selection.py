"""
Final model selection with a rule fixed BEFORE looking at the test season (pro VCT 2026): logistic regression,
tree ensembles and neural networks, on every training-data option, for the pre-match, live and round models.

Why: earlier runs compared algorithms on 2026 and the best-looking one was picked from those results. Choosing by
the test score turns the test season into part of the tuning, so its result is no longer an unbiased estimate.
Here every choice is made on a VALIDATION set inside the training period, and 2026 is used once, at the end.

THE RULE (pre-registered; printed at the start of every run)
  1. Validation set = the latest 20% of the pro 2023-2025 maps (by Game ID = time order; for live/round, of the maps
     with round data). The same validation maps are used for every candidate, whatever its training data.
  2. Every candidate (training data option x algorithm x feature set x setting) is trained on its training data
     WITHOUT the validation maps, and scored on them (log loss). Networks stop early on the same validation maps.
  3. The 1-standard-error rule: take the candidate with the lowest validation log loss; then choose the SIMPLEST
     candidate whose validation log loss is within one standard error of it (standard error of the paired
     difference, by bootstrap over validation maps). Simplest = logistic regression < tree ensembles < neural
     networks; then pro 2023-2025 only < added data. This is the standard way to avoid picking a model for a lead
     that is within noise.
  4. The best setting of every option x algorithm x feature set is retrained on all its training data (validation maps included;
     networks for the number of epochs early stopping chose) and tested ONCE on pro 2026. The test only REPORTS how
     they did; it does not change the choice. Verdicts vs logistic regression on pro 2023-2025 (the current pro
     model): paired bootstrap 95% CI by map, before and after Holm. Random algorithms (trees, networks) are refitted
     with 3 seeds; the table shows the range and in how many seeds each verdict holds.

Training data options:
  B   pro 2023-2025 (the current pro model)
  D   B + pro-level 2021-2022 (Challengers main events, LCQs, Masters, Champions) at full weight      [sheets D1/D2]
  D½  B + pro-level 2021-2022 at half weight
  C   B + 2021-2022 open and closed qualifiers at half weight (live/round)       [option C of sheets I1/I2]
  A   2021-2022 open and closed qualifiers ONLY, no pro data (live/round; only with --options A, set aside for now)
  E   ALL the data: pro 2023-2025 + pro-level 2021-2022 + qualifiers 2021-2022, all at full weight
  Eq½ pro 2023-2025 + pro-level 2021-2022 at full weight + qualifiers at half weight
  E½  pro 2023-2025 at full weight + pro-level 2021-2022 and qualifiers both at half weight
  Cf  pro 2023-2025 + qualifiers 2021-2022 at full weight
  Eo½ pro 2023-2025 + pro-level 2021-2022 at half weight + qualifiers at full weight
  O   pro-level 2021-2022 ONLY (no 2023-2025): separates the effect of the era from the effect of the level
  OQ  pro-level 2021-2022 + qualifiers 2021-2022, no 2023-2025 (all the old data, nothing recent)
  Progressive (recency) weights: each season counts a fixed fraction of the season after it,
  weight = rate ^ (2025 - season): rate 3/4 gives 2025 1, 2024 0.75, 2023 0.56, 2022 0.42, 2021 0.32;
  rate 1/2 gives 2025 1, 2024 0.5, 2023 0.25, 2022 0.13, 2021 0.06.
  R¾  all pro 2021-2025 (pro-level 2021-22 + pro 2023-25), rate 3/4
  R½  all pro 2021-2025, rate 1/2
  RQ¾ all pro 2021-2025 + qualifiers 2021-22, rate 3/4
  RQ½ all pro 2021-2025 + qualifiers 2021-22, rate 1/2
  (2026 is never in training: it is the test season.)
  (B is option B of sheets I1/I2, so live/round cover every training set of D1, D2, I1 and I2)

Algorithms (codes for --algos):
  LR  logistic regression          ET  Extra Trees           RF  Random Forest
  MLP  RNN  GRU  LSTM  CNN  CNNLSTM  TRANSFORMER  neural networks (as in test_deep_learning.py)
  FT  FT-Transformer (pre-match only)
  Feature sets: pre-match 4 final | 11 match features (networks: each team's last 10 maps + the 4 final features;
  MLP the 4 final features; FT-Transformer the 11).
  Live/round: engineered (hand-made combinations such as lead x rounds left) | rich (engineered + raw values) |
  raw (ONLY the raw state: Elo gap, both scores, round number, side, both buy types; no hand-made combinations, so
  every algorithm has to find the patterns itself). Networks read the rounds of a map in order with rich and raw
  inputs (--net-inputs rich or raw for one only, which halves their time); the MLP sees one round at a time.
  Network settings: pre-match 16 per network (hidden 32/64 x layers 1/2 x dropout 0/0.3 x learning rate 0.001/0.003);
  live/round 8 per network (one layer; the round sequences make two layers too slow).

Long runs can be split: every run saves its results per model and algorithm in figures/final_selection_cache/, and
the report at the end of every run (or --report alone) applies the rule to EVERYTHING saved so far for that model.
Run from the valorant-fyp folder (needs PyTorch for the networks):
    python final_selection.py --quick                                 check that everything works (minutes)
    python final_selection.py --part prematch                         all algorithms, pre-match (about 2-6 hours)
    python final_selection.py --part live --algos LR,ET,RF            classical algorithms first (a few hours)
    python final_selection.py --part live --algos RNN,GRU,LSTM        then networks: roughly 1-2 hours per network
    python final_selection.py --part live --algos MLP,CNN,CNNLSTM,TRANSFORMER
    (the same for --part round)
    python final_selection.py --part live --algos LR,ET,RF --options E,Eqh,Eh   only some data options
                                                                      (h = ½; results are added to what is saved)
    python final_selection.py --part live --report                    report only, from what is saved
Writes figures/final_selection_<part>.csv (every finalist: validation and test results, verdicts, seeds) and
       figures/final_selection_<part>_stages.csv (accuracy by stage of the map, live/round)
"""
import glob
import os
import pickle
import sys
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from round_model import (MAP_FEATURES, PRO_TIERS, ROUND_FEATURES, SEMI_TIERS, build_rounds, state_features,
                         tier_ids)

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
REPORT_ONLY = "--report" in sys.argv or "--research" in sys.argv
RESEARCH_ONLY = "--research" in sys.argv          # the research report only (non-linear models, input sets apart)
arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
PARTS = arg("--part", "prematch,live,round").split(",")
CODES = {"LR": "Logistic Regression", "ET": "Extra Trees", "RF": "Random Forest", "MLP": "MLP", "RNN": "RNN",
         "GRU": "GRU", "LSTM": "LSTM", "CNN": "CNN-1D", "CNNLSTM": "CNN+LSTM", "TRANSFORMER": "Transformer",
         "FT": "FT-Transformer"}
ALG_LIST = [CODES[c.strip().upper()] for c in arg("--algos", ",".join(CODES)).split(",")]
CLASSICAL = ["Logistic Regression", "Extra Trees", "Random Forest"]
NETS = [a for a in ALG_LIST if a not in CLASSICAL]
SEEDS = [0] if QUICK else [0, 1, 2]
NET_INPUTS = arg("--net-inputs", "rich,raw").split(",")
if REPORT_ONLY:
    NETS = []                                              # a report only reads saved results: no networks to build
OPT_ARG = arg("--options", None)                           # e.g. --options E,Eq½,E½ to run only some data options      # live/round networks: which input sets to try
if NETS and not REPORT_ONLY:
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except ImportError:
        print("PyTorch is not installed: the networks are skipped (pip install torch to include them).")
        NETS = []
os.makedirs("figures/final_selection_cache", exist_ok=True)
SFX = "_quick" if QUICK else ""
VAL_SHARE = 0.2
N_BOOT = 300 if QUICK else 1000
FINAL = ["elo_tuned", "player_diff", "fam_diff", "map_pick"]
ELEVEN = FINAL + ["comp_diff", "teammap_diff", "form_diff", "h2h_diff", "comfort_diff", "elo_x_intl", "elo_x_playoff"]
COMPLEXITY = {"Logistic Regression": 0, "Extra Trees": 1, "Random Forest": 1}       # every network: 2
OPT_ORDER = {"B": 0, "D": 1, "D½": 1, "C": 1, "A": 1, "E": 1, "Eq½": 1, "E½": 1, "Cf": 1, "Eo½": 1, "O": 1, "OQ": 1, "R¾": 1, "R½": 1, "RQ¾": 1, "RQ½": 1}
OPT_LABEL = {"B": "pro 2023-2025", "D": "+ pro-level 2021-22 (full weight)", "D½": "+ pro-level 2021-22 (half weight)",
             "C": "+ qualifiers 2021-22 (half weight)", "A": "qualifiers 2021-22 only",
             "E": "+ pro-level and qualifiers 2021-22 (full weight)",
             "Eq½": "+ pro-level 2021-22 (full) + qualifiers (half)",
             "E½": "+ pro-level and qualifiers 2021-22 (half weight)",
             "Cf": "+ qualifiers 2021-22 (full weight)", "Eo½": "+ pro-level 2021-22 (half) + qualifiers (full)",
             "O": "pro-level 2021-22 only", "OQ": "pro-level + qualifiers 2021-22 only (no 2023-25)",
             "R¾": "all pro 2021-25, recent seasons weighted more (rate 3/4)",
             "R½": "all pro 2021-25, recent seasons weighted more (rate 1/2)",
             "RQ¾": "all pro 2021-25 + qualifiers, recent weighted more (rate 3/4)",
             "RQ½": "all pro 2021-25 + qualifiers, recent weighted more (rate 1/2)"}
TYP = {"Eco": 3000, "Semi-eco": 8000, "Semi-buy": 15000, "Full buy": 22000}
BUY = {"Eco": 0, "Semi-eco": 1, "Semi-buy": 2, "Full buy": 3}
STAGES = [("Before round 1", 1, 1), ("Rounds 2-6", 2, 6), ("Rounds 7-12", 7, 12), ("Rounds 13-18", 13, 18),
          ("Rounds 19-24", 19, 24), ("Overtime", 25, 99)]

S = lambda m: make_pipeline(StandardScaler(), m)
TREE_GRID = (dict(max_depth=[8], min_samples_leaf=[50]) if QUICK else
             dict(max_depth=[12, None], min_samples_leaf=[20, 100]))
ALGOS = {
    "Logistic Regression": (lambda p, s: S(LogisticRegression(max_iter=3000, **p)),
                            dict(C=[0.1]) if QUICK else dict(C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])),
    "Extra Trees": (lambda p, s: ExtraTreesClassifier(n_estimators=100 if QUICK else 300, max_features="sqrt",
                                                      n_jobs=-1, random_state=s, **p), TREE_GRID),
    "Random Forest": (lambda p, s: RandomForestClassifier(n_estimators=100 if QUICK else 300, max_features=0.5,
                                                          n_jobs=-1, random_state=s, **p), TREE_GRID),
}
NET_GRID_PRE = (dict(hidden=[32], layers=[1], dropout=[0.2], lr=[3e-3]) if QUICK else
                dict(hidden=[32, 64], layers=[1, 2], dropout=[0.0, 0.3], lr=[1e-3, 3e-3]))
NET_GRID_SEQ = (dict(hidden=[32], layers=[1], dropout=[0.2], lr=[3e-3]) if QUICK else
                dict(hidden=[32, 64], layers=[1], dropout=[0.0, 0.3], lr=[1e-3, 3e-3]))
MAX_EPOCHS, PATIENCE, BATCH = (6, 2, 64) if QUICK else (80, 6, 64)

print("THE RULE" + __doc__.split("THE RULE")[1].split("Training data options")[0])
data = pd.read_csv("data/maps_dataset_features.csv").sort_values("Game ID").reset_index(drop=True)
OLD_PRO = tier_ids(data, [2021, 2022], PRO_TIERS)
QUAL = tier_ids(data, [2021, 2022], SEMI_TIERS)
new_ids = sorted(data.loc[data["year"].between(2023, 2025), "Game ID"])
TEST_IDS = set(data.loc[data["year"] == 2026, "Game ID"])
GID_YEAR = data.drop_duplicates("Game ID").set_index("Game ID")["year"]
OPTIONS = {"B": {}, "D": {"old": [(OLD_PRO, 1.0)]}, "D½": {"old": [(OLD_PRO, 0.5)]}, "C": {"old": [(QUAL, 0.5)]},
           "A": {"only": [(QUAL, 1.0)]}, "O": {"only": [(OLD_PRO, 1.0)]},
           "OQ": {"only": [(OLD_PRO, 1.0), (QUAL, 1.0)]}, "Cf": {"old": [(QUAL, 1.0)]},
           "Eo½": {"old": [(OLD_PRO, 0.5), (QUAL, 1.0)]},
           "R¾": {"decay": (0.75, False)}, "R½": {"decay": (0.5, False)},
           "RQ¾": {"decay": (0.75, True)}, "RQ½": {"decay": (0.5, True)}, "E": {"old": [(OLD_PRO, 1.0), (QUAL, 1.0)]},
           "Eq½": {"old": [(OLD_PRO, 1.0), (QUAL, 0.5)]}, "E½": {"old": [(OLD_PRO, 0.5), (QUAL, 0.5)]}}
VAL_IDS = set()
PRO_2326 = set(new_ids) | set(TEST_IDS)                    # all pro maps 2023-2026 (for the ranked model)
# ---- ranked (app) model: tested on the latest 20% of the 2021-22 qualifier maps (the closest data to ranked play),
# validation = the 20% before them, training = qualifiers before that (+ pro data in some options). No Elo.
RANKED_OPTS = ["Q", "QP½", "QP", "QA½", "QA", "PA"]
OPTIONS.update({"Q": {"only": [(QUAL, 1.0)]},
                "QP½": {"only": [(QUAL, 1.0), (PRO_2326, 0.5)]}, "QP": {"only": [(QUAL, 1.0), (PRO_2326, 1.0)]},
                "QA½": {"only": [(QUAL, 1.0), (PRO_2326 | OLD_PRO, 0.5)]},
                "QA": {"only": [(QUAL, 1.0), (PRO_2326 | OLD_PRO, 1.0)]},
                "PA": {"only": [(PRO_2326 | OLD_PRO, 1.0)]}})
OPT_LABEL.update({"Q": "qualifiers only (earlier 60%)", "QP½": "qualifiers + pro 2023-26 (half weight)",
                  "QP": "qualifiers + pro 2023-26 (full weight)", "QA½": "qualifiers + all pro 2021-26 (half weight)",
                  "QA": "qualifiers + all pro 2021-26 (full weight)", "PA": "all pro 2021-26 only, no qualifiers"})
OPT_ORDER.update({"Q": 0, "QP½": 1, "QP": 1, "QA½": 1, "QA": 1, "PA": 1})


def set_val(available):
    """Validation maps = the latest 20% of the pro 2023-2025 maps that this model can use (live/round: maps with
    round data, so the validation set is not empty where round data is missing, e.g. most of 2025)."""
    global VAL_IDS
    ids = sorted(set(available) & set(new_ids))
    VAL_IDS = set(ids[int(len(ids) * (1 - VAL_SHARE)):])


def weights(gids, opt, include_val):
    """Row weights of a training-data option (0 = not used)."""
    gids = np.asarray(gids)
    if "decay" in OPTIONS[opt]:                             # progressive weights by season
        rate, with_q = OPTIONS[opt]["decay"]
        used = np.isin(gids, new_ids) | np.isin(gids, list(OLD_PRO | (QUAL if with_q else set())))
        yr = pd.Series(gids).map(GID_YEAR).fillna(2026).to_numpy()
        w = np.where(used, rate ** (2025 - yr), 0.0)
        if not include_val:
            w[np.isin(gids, list(VAL_IDS))] = 0.0
        return w
    if "only" in OPTIONS[opt]:                              # options without any 2023-2025 rows / ranked options
        w = np.zeros(len(gids))
        for ids, wt in OPTIONS[opt]["only"]:
            w[np.isin(gids, list(ids))] = wt
        if not include_val:
            w[np.isin(gids, list(VAL_IDS))] = 0.0
        w[np.isin(gids, list(TEST_IDS))] = 0.0             # the test maps are never trained on
        return w
    w = np.where(np.isin(gids, new_ids), 1.0, 0.0)
    if not include_val:
        w[np.isin(gids, list(VAL_IDS))] = 0.0
    for ids, wt in OPTIONS[opt].get("old", []):
        w[np.isin(gids, list(ids))] = wt
    return w


def proba(m, X):
    return np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def row_ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_map(gids, losses):
    s = pd.Series(losses).groupby(np.asarray(gids)).agg(["sum", "size"]).sort_index()
    return s["sum"].to_numpy(), s["size"].to_numpy()


def fit_w(m, X, y, w):
    if hasattr(m, "steps"):
        return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m.fit(X, y, sample_weight=w)


def paired(base_sum, new_sum, cnt, rng):
    """Bootstrap by map of the mean log-loss gain (positive = new model better)."""
    k = len(cnt)
    g = np.array([(base_sum[i].sum() - new_sum[i].sum()) / cnt[i].sum()
                  for i in (rng.integers(0, k, k) for _ in range(N_BOOT))])
    lo, hi = np.percentile(g, [2.5, 97.5])
    p = min(1.0, 2 * min((g <= 0).mean(), (g >= 0).mean()) + 1 / len(g))
    return (base_sum.sum() - new_sum.sum()) / cnt.sum(), lo, hi, p, g.std()


def holm(rows):
    tested = sorted([r for r in rows if r.get("p_value") is not None], key=lambda r: r["p_value"])
    m, still = len(tested), True
    for k, r in enumerate(tested):
        still = still and r["p_value"] <= 0.05 / (m - k)
        r["verdict_holm"] = r["verdict"] if still and r["verdict"] != "same" else "same"


def cache_path(part, algo, opts):
    tag = "-".join(o.replace("½", "h").replace("¾", "q3") for o in opts)
    return f"figures/final_selection_cache/{part}{SFX}_{algo.replace(' ', '_').replace('+', 'plus')}_{tag}.pkl"


def pick_opts(default):
    """--options codes: h = ½ and q3 = ¾ (easy to type), e.g. Dh, Eqh, Rq3, RQh."""
    if not OPT_ARG:
        return default
    want = [x.strip().replace("q3", "¾").replace("Q3", "¾").replace("h", "½") for x in OPT_ARG.split(",")]
    return [o for o in want if o in default]


def finalize(part, algo, cands, meta):
    """Keep the best setting per option x feature set (finalists), refit each on all its training data with every
    seed, save."""
    best = {}
    for c in cands:
        c["val_log_loss"] = row_ll(meta["y_val"], c["val_pred"]).mean()
        k = (c["option"], c["features"])                   # one finalist per option x feature set
        if k not in best or c["val_log_loss"] < best[k]["val_log_loss"]:
            best[k] = c
    fin = []
    for c in best.values():
        seeds = [0] if algo == "Logistic Regression" else SEEDS
        test_preds = {s: c["refit"](s) for s in seeds}
        fin.append({k: v for k, v in c.items() if k != "refit"} | {"test_preds": test_preds})
    with open(cache_path(part, algo, sorted({c["option"] for c in cands})), "wb") as f:
        pickle.dump({"meta": meta, "finalists": fin}, f)


# =================================================================== neural networks
if NETS:
    class CausalConv(nn.Module):
        def __init__(self, cin, cout, k=3):
            super().__init__()
            self.pad, self.conv = k - 1, nn.Conv1d(cin, cout, k)

        def forward(self, x):
            return torch.relu(self.conv(nn.functional.pad(x.transpose(1, 2), (self.pad, 0)))).transpose(1, 2)

    class Encoder(nn.Module):
        """(batch, time, features) -> (batch, time, hidden); every type uses only the current and earlier steps."""
        def __init__(self, arch, nin, hidden, layers, dropout, max_len=64):
            super().__init__()
            self.arch, drop = arch, (dropout if layers > 1 else 0.0)
            if arch in ("RNN", "LSTM", "GRU"):
                self.net = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[arch](nin, hidden, num_layers=layers,
                                                                              batch_first=True, dropout=drop)
            elif arch == "CNN-1D":
                self.net = nn.Sequential(*[CausalConv(nin if i == 0 else hidden, hidden) for i in range(layers + 1)])
            elif arch == "CNN+LSTM":
                self.conv = CausalConv(nin, hidden)
                self.net = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=drop)
            elif arch == "Transformer":
                self.proj, self.pos = nn.Linear(nin, hidden), nn.Embedding(max_len, hidden)
                self.net = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                                                            dropout=dropout, batch_first=True), layers)
            else:                                               # MLP: each step on its own
                mods, c = [], nin
                for _ in range(layers):
                    mods += [nn.Linear(c, hidden), nn.ReLU(), nn.Dropout(dropout)]; c = hidden
                self.net = nn.Sequential(*mods)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            if self.arch in ("RNN", "LSTM", "GRU"):
                out, _ = self.net(x)
            elif self.arch == "CNN+LSTM":
                out, _ = self.net(self.conv(x))
            elif self.arch == "Transformer":
                T = x.shape[1]
                h = self.proj(x) + self.pos(torch.arange(T))[None]
                out = self.net(h, mask=torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
            else:
                out = self.net(x)
            return self.drop(out)

    class FTTransformer(nn.Module):
        def __init__(self, nin, hidden, layers, dropout):
            super().__init__()
            self.w, self.b = nn.Parameter(torch.randn(nin, hidden) * 0.1), nn.Parameter(torch.zeros(nin, hidden))
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            self.net = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                                                        dropout=dropout, batch_first=True), layers)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            tok = x[:, :, None] * self.w[None] + self.b[None]
            return self.head(self.net(torch.cat([self.cls.expand(len(x), -1, -1), tok], 1))[:, 0]).squeeze(-1)

    class SeqNet(nn.Module):                                  # live / round: one output per round
        def __init__(self, arch, nin, hidden, layers, dropout, max_len):
            super().__init__()
            self.enc, self.head = Encoder(arch, nin, hidden, layers, dropout, max_len), nn.Linear(hidden, 1)

        def forward(self, x):
            return self.head(self.enc(x)).squeeze(-1)

    class PreNet(nn.Module):                                  # pre-match
        def __init__(self, arch, hidden, layers, dropout, nseq, ntab4, ntab11):
            super().__init__()
            self.arch = arch
            if arch == "MLP":
                self.enc = Encoder("MLP", ntab4, hidden, layers, dropout)
                self.head = nn.Linear(hidden, 1)
            elif arch == "FT-Transformer":
                self.ft = FTTransformer(ntab11, hidden, layers, dropout)
            else:
                self.enc = Encoder(arch, nseq, hidden, layers, dropout, max_len=11)
                self.head = nn.Sequential(nn.Linear(hidden + ntab4, hidden), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(hidden, 1))

        def forward(self, sa, sb, t4, t11):
            if self.arch == "MLP":
                return self.head(self.enc(t4)).squeeze(-1)
            if self.arch == "FT-Transformer":
                return self.ft(t11)
            return self.head(torch.cat([self.enc(sa)[:, -1] - self.enc(sb)[:, -1], t4], 1)).squeeze(-1)

    def train_net(make, loss_fn, n_fit, n_val, lr, seed, epochs=None):
        """Early stopping on the validation maps (epochs=None), or a fixed number of epochs (final refit)."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = make()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        rng, best, best_state, wait, used = np.random.default_rng(seed), np.inf, None, 0, 1
        for ep in range(epochs or MAX_EPOCHS):
            model.train()
            idx = rng.permutation(n_fit)
            for i in range(0, n_fit, BATCH):
                loss = loss_fn(model, "fit", idx[i:i + BATCH])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if epochs:
                continue
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model, "val", np.arange(n_val)).item()
            if vl < best - 1e-5:
                best, wait, used = vl, 0, ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if not epochs:
            model.load_state_dict(best_state)
        model.eval()
        return model, used


# =================================================================== the report: rule + one test
def report(part):
    files = sorted(glob.glob(f"figures/final_selection_cache/{part}{SFX}_*.pkl"), key=os.path.getmtime)
    files = [f for f in files if (SFX == "_quick") == ("_quick_" in os.path.basename(f))]
    if not files:
        print(f"  nothing saved yet for {part}")
        return
    found, meta = {}, None
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        meta = meta or d["meta"]
        if len(d["meta"]["y_val"]) != len(meta["y_val"]) or len(d["meta"]["y_te"]) != len(meta["y_te"]):
            print(f"  skipped {f}: made from different data (re-run it)")
            continue
        for c in d["finalists"]:                               # a later run of the same candidate replaces it
            found[(c["option"], c["algorithm"], c["features"])] = c
    fin = list(found.values())
    y_val, g_val, y_te, g_te = meta["y_val"], meta["g_val"], meta["y_te"], meta["g_te"]
    algos_in = sorted({c["algorithm"] for c in fin})
    print(f"\n  REPORT {part}: algorithms saved so far: {', '.join(algos_in)}")
    rng = np.random.default_rng(0)
    best = min(fin, key=lambda c: c["val_log_loss"])
    b_sum, cnt = per_map(g_val, row_ll(y_val, best["val_pred"]))
    for c in fin:
        c_sum, _ = per_map(g_val, row_ll(y_val, c["val_pred"]))
        gain, _, _, _, se = paired(c_sum, b_sum, cnt, rng)
        c["behind_best"], c["se"], c["within_1se"] = gain, se, gain <= se
    chosen = min([c for c in fin if c["within_1se"]],
                 key=lambda c: (COMPLEXITY.get(c["algorithm"], 2), OPT_ORDER[c["option"]], c["val_log_loss"]))
    print(f"  Validation ({len(set(g_val)):,} maps, latest 20% of pro 2023-2025):")
    for c in sorted(fin, key=lambda c: c["val_log_loss"]):
        print(f"    {c['option']:3s} {c['algorithm']:20s} {c['features']:15s} val log loss {c['val_log_loss']:.4f}  "
              f"behind best {c['behind_best']:+.4f} (1 SE {c['se']:.4f}) {'within 1 SE' if c['within_1se'] else ''}")
    print(f"  RULE -> chosen BEFORE testing: {chosen['algorithm']}, option {chosen['option']} ({OPT_LABEL[chosen['option']]}),"
          f" {chosen['features']} features, {chosen['params']}")
    if not any(c["algorithm"] == "Logistic Regression" and c["option"] == "B" for c in fin):
        print("  (logistic regression is not saved yet: run --algos LR for the test verdicts)")
        return
    ref = min([c for c in fin if c["algorithm"] == "Logistic Regression" and c["option"] == "B"],
              key=lambda c: c["val_log_loss"])           # its better feature set, chosen on validation
    r_sum, r_cnt = per_map(g_te, row_ll(y_te, ref["test_preds"][0]))
    rows = []
    for c in fin:
        lls = [row_ll(y_te, p).mean() for p in c["test_preds"].values()]
        row = {"option": c["option"], "training data": OPT_LABEL[c["option"]], "algorithm": c["algorithm"],
               "features": c["features"], "params": str(c["params"]), "val_log_loss": c["val_log_loss"],
               "within_1se": c["within_1se"], "chosen_by_rule": c is chosen, "test_log_loss": lls[0],
               "test_ll_min": min(lls), "test_ll_max": max(lls),
               "test_accuracy": ((c["test_preds"][0] > 0.5) == y_te).mean(), "seeds": len(lls)}
        if c is ref:
            row.update(verdict="baseline", verdict_holm="baseline", seeds_agree="")
        else:
            vs = []
            for s, p in c["test_preds"].items():
                c_sum, _ = per_map(g_te, row_ll(y_te, p))
                gain, lo, hi, pv, _ = paired(r_sum, c_sum, r_cnt, rng)
                vs.append("better" if lo > 0 else ("worse" if hi < 0 else "same"))
                if s == 0:
                    row.update(gain=gain, ci_low=lo, ci_high=hi, p_value=pv, verdict=vs[-1])
            row["seeds_agree"] = f"{vs.count(row['verdict'])} of {len(vs)}"
        rows.append(row)
    holm(rows)
    out = pd.DataFrame(rows).sort_values("test_log_loss")
    print(f"\n  Test on pro 2026 ({len(set(g_te)):,} maps), once; reference = logistic regression, option B")
    print(out[["option", "algorithm", "features", "val_log_loss", "chosen_by_rule", "test_log_loss", "test_ll_min",
               "test_ll_max", "test_accuracy", "verdict", "verdict_holm", "seeds_agree"]].round(4).to_string(index=False))
    print("  verdict: paired bootstrap by map (seed 0); seeds_agree: in how many seeds the same verdict holds;\n"
          "  test_ll_min/max: range over the seeds.")
    out.to_csv(f"figures/final_selection_{part}{SFX}.csv", index=False)
    if meta.get("rnd_te") is not None:
        rnd, st = meta["rnd_te"], []
        for c in fin:
            p = c["test_preds"][0]
            d = {"option": c["option"], "algorithm": c["algorithm"], "features": c["features"]}
            for name, a, b in STAGES:
                sel = (rnd >= a) & (rnd <= b)
                d[name] = round(((p[sel] > 0.5) == y_te[sel]).mean() * 100, 1) if sel.any() else np.nan
            st.append(d)
        st = pd.DataFrame(st).sort_values(["algorithm", "option"])
        print("\n  Accuracy (%) by stage of the map, pro 2026 (seed 0):")
        print(st.to_string(index=False))
        st.to_csv(f"figures/final_selection_{part}{SFX}_stages.csv", index=False)


# =================================================================== the research report
def report_research(part):
    """Non-linear models only (no logistic regression), each input set judged on its own.
    Within each input set: the rule (lowest validation log loss; the simplest model within one standard error:
    trees < networks, then the reference data < added data), and the plain lowest-validation model, both shown.
    Test verdicts (pro 2026 / ranked: latest qualifiers), paired bootstrap by map, before and after Holm:
      data question:  each data option vs the SAME algorithm and input set trained on the reference data
                      (pro: B = pro 2023-2025; ranked: Q = qualifiers only)
      model question: each model vs the rule's choice in its input set"""
    files = sorted(glob.glob(f"figures/final_selection_cache/{part}{SFX}_*.pkl"), key=os.path.getmtime)
    files = [f for f in files if (SFX == "_quick") == ("_quick_" in os.path.basename(f))]
    found, meta = {}, None
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        meta = meta or d["meta"]
        if len(d["meta"]["y_val"]) != len(meta["y_val"]) or len(d["meta"]["y_te"]) != len(meta["y_te"]):
            continue
        for c in d["finalists"]:
            found[(c["option"], c["algorithm"], c["features"])] = c
    fin = [c for c in found.values() if c["algorithm"] != "Logistic Regression"]
    if not fin:
        print(f"  research report {part}: no non-linear results saved yet")
        return
    ref_opt = "Q" if part.startswith("ranked") else "B"
    y_val, g_val, y_te, g_te = meta["y_val"], meta["g_val"], meta["y_te"], meta["g_te"]
    rng = np.random.default_rng(0)
    group = lambda c: c["features"].split(",")[0].strip()
    test_name = "the latest 20% of the 2021-22 qualifier maps" if part.startswith("ranked") else "pro 2026"
    print(f"\n  RESEARCH REPORT {part}: non-linear models only, each input set separately; test = {test_name}")
    print(f"  algorithms saved: {', '.join(sorted({c['algorithm'] for c in fin}))}; reference data = option {ref_opt} "
          f"({OPT_LABEL[ref_opt]})")
    rows, chosen_rows = [], []
    for g in sorted({group(c) for c in fin}):
        cs = [c for c in fin if group(c) == g]
        best = min(cs, key=lambda c: c["val_log_loss"])
        b_sum, cnt = per_map(g_val, row_ll(y_val, best["val_pred"]))
        for c in cs:
            c_sum, _ = per_map(g_val, row_ll(y_val, c["val_pred"]))
            gain, _, _, _, se = paired(c_sum, b_sum, cnt, rng)
            c["r_within"] = gain <= se
        chosen = min([c for c in cs if c["r_within"]],
                     key=lambda c: (COMPLEXITY.get(c["algorithm"], 2), 0 if c["option"] == ref_opt else 1,
                                    c["val_log_loss"]))
        print(f"\n  ---- input set: {g} ({len(cs)} finalists)")
        print(f"  lowest validation log loss: {best['algorithm']}, option {best['option']} ({OPT_LABEL[best['option']]}), "
              f"{best['val_log_loss']:.4f}")
        print(f"  rule (simplest within 1 SE): {chosen['algorithm']}, option {chosen['option']} "
              f"({OPT_LABEL[chosen['option']]}), {chosen['val_log_loss']:.4f}")
        ch_sum = {s: per_map(g_te, row_ll(y_te, p))[0] for s, p in chosen["test_preds"].items()}
        grp = []
        for c in cs:
            lls = [row_ll(y_te, p).mean() for p in c["test_preds"].values()]
            row = {"input set": g, "option": c["option"], "training data": OPT_LABEL[c["option"]],
                   "algorithm": c["algorithm"], "features": c["features"], "params": str(c["params"]),
                   "val_log_loss": c["val_log_loss"], "lowest_val": c is best, "chosen_by_rule": c is chosen,
                   "test_log_loss": lls[0], "test_ll_min": min(lls), "test_ll_max": max(lls),
                   "test_accuracy": ((c["test_preds"][0] > 0.5) == y_te).mean()}
            ref = found.get((ref_opt, c["algorithm"], c["features"]))
            for key, base in (("data", ref), ("model", chosen)):
                if base is None or base is c:
                    row[f"{key}_verdict"] = "reference" if base is c else "no reference"
                    continue
                vs = []
                for s, p in c["test_preds"].items():
                    bp = base["test_preds"].get(s, base["test_preds"][0])
                    b_s, b_c = per_map(g_te, row_ll(y_te, bp))
                    gain, lo, hi, pv, _ = paired(b_s, per_map(g_te, row_ll(y_te, p))[0], b_c, rng)
                    vs.append("better" if lo > 0 else ("worse" if hi < 0 else "same"))
                    if s == 0:
                        row.update({f"{key}_gain": gain, f"{key}_ci_low": lo, f"{key}_ci_high": hi,
                                    f"{key}_p": pv, f"{key}_verdict": vs[-1]})
                row[f"{key}_seeds_agree"] = f"{vs.count(row[key + '_verdict'])} of {len(vs)}"
            grp.append(row)
        for key in ("data", "model"):                       # Holm within the input set, per question
            tested = sorted([r for r in grp if r.get(f"{key}_p") is not None], key=lambda r: r[f"{key}_p"])
            still = True
            for k, r in enumerate(tested):
                still = still and r[f"{key}_p"] <= 0.05 / (len(tested) - k)
                r[f"{key}_verdict_holm"] = r[f"{key}_verdict"] if still and r[f"{key}_verdict"] != "same" else "same"
            for r in grp:
                r.setdefault(f"{key}_verdict_holm", r.get(f"{key}_verdict"))
        out = pd.DataFrame(grp).sort_values("test_log_loss")
        cols = ["option", "algorithm", "val_log_loss", "lowest_val", "chosen_by_rule", "test_log_loss", "test_ll_min",
                "test_ll_max", "test_accuracy", "data_verdict", "data_verdict_holm", "model_verdict", "model_verdict_holm"]
        print(out[cols].round(4).to_string(index=False))
        rows += grp
        cr = next(r for r in grp if r["chosen_by_rule"])
        br = next(r for r in grp if r["lowest_val"])
        chosen_rows.append({"input set": g, "rule's choice": f"{cr['algorithm']}, {cr['option']}",
                            "val": cr["val_log_loss"], "test log loss": cr["test_log_loss"],
                            "test accuracy": cr["test_accuracy"], "lowest validation": f"{br['algorithm']}, {br['option']}",
                            "its test log loss": br["test_log_loss"]})
    print(f"\n  The three input sets side by side (each one's own choice; {test_name}):")
    print(pd.DataFrame(chosen_rows).round(4).to_string(index=False))
    print("  data_verdict: vs the same algorithm and input set trained on the reference data; model_verdict: vs the "
          "rule's choice in that input set.\n  Verdicts use seed 0; seeds_agree columns in the CSV give the other seeds.")
    pd.DataFrame(rows).to_csv(f"figures/research_{part}{SFX}.csv", index=False)
    if meta.get("rnd_te") is not None:
        rnd, st = meta["rnd_te"], []
        for c in fin:
            p = c["test_preds"][0]
            d = {"input set": group(c), "option": c["option"], "algorithm": c["algorithm"]}
            for name, a, b in STAGES:
                sel = (rnd >= a) & (rnd <= b)
                d[name] = round(((p[sel] > 0.5) == y_te[sel]).mean() * 100, 1) if sel.any() else np.nan
            st.append(d)
        pd.DataFrame(st).to_csv(f"figures/research_{part}{SFX}_stages.csv", index=False)
    print(f"  Saved figures/research_{part}{SFX}.csv" + (" and _stages.csv" if meta.get("rnd_te") is not None else ""))


# =================================================================== part 1: pre-match
def run_prematch():
    print("\n================ PRE-MATCH (map winner before the map)")
    d = data
    set_val(new_ids)
    gids = d["Game ID"].to_numpy()
    y = d["A_won"].to_numpy().astype(float)
    te, val = np.isin(gids, list(TEST_IDS)), np.isin(gids, list(VAL_IDS))
    meta = dict(y_val=y[val], g_val=gids[val], y_te=y[te], g_te=gids[te], rnd_te=None)
    opts = pick_opts(["B", "D", "D½", "C", "Cf", "E", "Eq½", "E½", "Eo½", "O", "OQ", "R¾", "R½", "RQ¾", "RQ½"])
    for o in opts:
        print(f"  option {o}: {(weights(gids, o, False) > 0).sum():,} maps for fitting, "
              f"{(weights(gids, o, True) > 0).sum():,} for the final refit ({OPT_LABEL[o]})")
    for algo in [a for a in ALG_LIST if a in CLASSICAL]:
        t0, cands = time.time(), []
        make, grid = ALGOS[algo]
        for opt in opts:
            w_fit, w_all = weights(gids, opt, False), weights(gids, opt, True)
            fit, allm = w_fit > 0, w_all > 0
            for fs, cols in {"4 final": FINAL, "11 match": ELEVEN}.items():
                X = d[cols].fillna(0).to_numpy()
                for params in ParameterGrid(grid):
                    m = fit_w(make(params, 0), X[fit], y[fit], w_fit[fit])

                    def refit(s, params=params, X=X, allm=allm, w_all=w_all):
                        return proba(fit_w(make(params, s), X[allm], y[allm], w_all[allm]), X[te])
                    cands.append(dict(option=opt, algorithm=algo, features=fs, params=params,
                                      val_pred=proba(m, X[val]), refit=refit))
        finalize("prematch", algo, cands, meta)
        print(f"  {algo:20s} done ({time.time() - t0:.0f}s)")
    if NETS:
        prematch_nets(d, y, gids, val, te, opts, meta)
    report("prematch")
    report_research("prematch")


def prematch_nets(d, y, gids, val, te, opts, meta):
    """Each team's last 10 maps (won, strength gap, player form gap, familiarity, map pick, same map, present), as in
    test_deep_learning.py; history uses every earlier map in the data."""
    N, F = 10, 7
    hist, seqA, seqB = {}, [], []
    for ta, tb, mp, won, e, pdf, fam, pick in zip(d["Team A"], d["Team B"], d["Map"], d["A_won"], *[d[c] for c in FINAL]):
        for t, out in ((ta, seqA), (tb, seqB)):
            h = list(hist.get(t, []))[-N:]
            s = np.zeros((N, F), np.float32)
            for i, (rec, m_) in enumerate(h):
                s[N - len(h) + i] = rec + [1.0 if m_ == mp else 0.0, 1.0]
            out.append(s)
        won, e, pdf, fam, pick = float(won), float(e), float(pdf), float(fam), float(pick)
        hist.setdefault(ta, deque(maxlen=N)).append(([won, e, pdf, fam, pick], mp))
        hist.setdefault(tb, deque(maxlen=N)).append(([1 - won, -e, -pdf, -fam, -pick], mp))
    SA, SB = torch.tensor(np.stack(seqA)), torch.tensor(np.stack(seqB))
    t4, t11 = d[FINAL].fillna(0).to_numpy(np.float32), d[ELEVEN].fillna(0).to_numpy(np.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    val_rows, te_rows = np.where(val)[0], np.where(te)[0]
    for arch in NETS:
        t0, cands = time.time(), []
        for opt in opts:
            w_fit, w_all = weights(gids, opt, False), weights(gids, opt, True)
            fit_rows, all_rows = np.where(w_fit > 0)[0], np.where(w_all > 0)[0]
            T4 = torch.tensor((t4 - t4[fit_rows].mean(0)) / (t4[fit_rows].std(0) + 1e-6))
            T11 = torch.tensor((t11 - t11[fit_rows].mean(0)) / (t11[fit_rows].std(0) + 1e-6))
            W = torch.tensor(w_all, dtype=torch.float32)
            run = lambda m, r: m(SA[r], SB[r], T4[r], T11[r])
            for g in ParameterGrid(NET_GRID_PRE):
                h, nl, dr, lr = int(g["hidden"]), int(g["layers"]), float(g["dropout"]), float(g["lr"])
                make = lambda h=h, nl=nl, dr=dr: PreNet(arch, h, nl, dr, F, len(FINAL), len(ELEVEN))

                def loss_fn(model, part, ix, fit_rows=fit_rows, W=W, run=run):
                    r = torch.as_tensor((fit_rows if part == "fit" else val_rows)[ix], dtype=torch.long)
                    l = nn.functional.binary_cross_entropy_with_logits(run(model, r), Y[r], reduction="none")
                    w = W[r] if part == "fit" else torch.ones(len(r))
                    return (l * w).sum() / w.sum()
                model, ep = train_net(make, loss_fn, len(fit_rows), len(val_rows), lr, 0)
                with torch.no_grad():
                    vp = torch.sigmoid(run(model, torch.as_tensor(val_rows, dtype=torch.long))).numpy()

                def refit(s, make=make, ep=ep, lr=lr, all_rows=all_rows, W=W, run=run):
                    def lf(model, part, ix):
                        r = torch.as_tensor(all_rows[ix], dtype=torch.long)
                        l = nn.functional.binary_cross_entropy_with_logits(run(model, r), Y[r], reduction="none")
                        return (l * W[r]).sum() / W[r].sum()
                    m, _ = train_net(make, lf, len(all_rows), 0, lr, s, epochs=ep)
                    with torch.no_grad():
                        return np.clip(torch.sigmoid(run(m, torch.as_tensor(te_rows, dtype=torch.long))).numpy(),
                                       1e-6, 1 - 1e-6)
                fs = {"MLP": "4 final", "FT-Transformer": "11 match"}.get(arch, "last 10 maps + 4")
                cands.append(dict(option=opt, algorithm=arch, features=fs, val_pred=np.clip(vp, 1e-6, 1 - 1e-6),
                                  params=dict(hidden=h, layers=nl, dropout=dr, lr=lr, epochs=ep), refit=refit))
        finalize("prematch", arch, cands, meta)
        print(f"  {arch:20s} done ({(time.time() - t0) / 60:.0f} min)")


# =================================================================== parts 2-3: live and round
ROUNDS = None


def get_rounds():
    global ROUNDS
    if ROUNDS is None:
        years = [y for y in range(2021, 2027) if os.path.exists(f"data/vct_{y}/matches/eco_rounds.csv")]
        print("  building round tables...")
        ROUNDS = pd.concat([build_rounds(data, [y], verbose=False) for y in years], ignore_index=True)
        ROUNDS = ROUNDS.sort_values(["Game ID", "Round Number"]).reset_index(drop=True)
    return ROUNDS


def run_rounds(part):
    target = "A_won" if part == "live" else "a_won_round"
    base = MAP_FEATURES if part == "live" else ROUND_FEATURES
    print(f"\n================ {'LIVE (map winner before each round)' if part == 'live' else 'ROUND (next-round winner)'}")
    R = get_rounds()
    keep = np.isin(R["Game ID"], new_ids) | np.isin(R["Game ID"], list(OLD_PRO | QUAL | TEST_IDS))
    R = R[keep].reset_index(drop=True)
    gids = R["Game ID"].to_numpy()
    set_val(gids)
    y = R[target].to_numpy().astype(float)
    E = state_features(R)[base]
    rich = E.copy()
    rich["score_a"], rich["score_b"] = R["sA"].astype(float), R["sB"].astype(float)
    rich["round_no"] = R["Round Number"].astype(float)
    rich["load_a_k"], rich["load_b_k"] = R["buy_a"].map(TYP).fillna(15000) / 1000, R["buy_b"].map(TYP).fillna(15000) / 1000
    rich["buy_a"], rich["buy_b"] = R["buy_a"].map(BUY).fillna(2), R["buy_b"].map(BUY).fillna(2)
    rich = rich.loc[:, ~rich.columns.duplicated()]
    raw = pd.DataFrame({"elo_tuned": R["elo_tuned"].astype(float), "score_a": R["sA"].astype(float),
                        "score_b": R["sB"].astype(float), "round_no": R["Round Number"].astype(float),
                        "a_attack": R["a_attack"].astype(float), "buy_a": R["buy_a"].map(BUY).fillna(2),
                        "buy_b": R["buy_b"].map(BUY).fillna(2)})
    fsets = {"engineered": E.to_numpy(), "rich": rich.to_numpy(), "raw": raw.to_numpy()}
    # option A (qualifiers only) is set aside for now; it runs only when asked for with --options A
    ALL_OPTS = ["B", "D", "D½", "C", "Cf", "E", "Eq½", "E½", "Eo½", "O", "OQ", "R¾", "R½", "RQ¾", "RQ½", "A"]
    opts = [o for o in ALL_OPTS if o != "A"] if not OPT_ARG else pick_opts(ALL_OPTS)
    fit_all(part, R, y, gids, fsets, opts)


def fit_all(part, R, y, gids, fsets, opts):
    te, val = np.isin(gids, list(TEST_IDS)), np.isin(gids, list(VAL_IDS))
    meta = dict(y_val=y[val], g_val=gids[val], y_te=y[te], g_te=gids[te], rnd_te=R.loc[te, "Round Number"].to_numpy())
    for o in opts:
        f = weights(gids, o, False) > 0
        print(f"  option {o}: {len(set(gids[f])):,} maps / {f.sum():,} rounds for fitting ({OPT_LABEL[o]})")
    for algo in [a for a in ALG_LIST if a in CLASSICAL]:
        t0, cands = time.time(), []
        make, grid = ALGOS[algo]
        for opt in opts:
            w_fit, w_all = weights(gids, opt, False), weights(gids, opt, True)
            fit, allm = w_fit > 0, w_all > 0
            for fs, X in fsets.items():
                for params in ParameterGrid(grid):
                    m = fit_w(make(params, 0), X[fit], y[fit], w_fit[fit])

                    def refit(s, params=params, X=X, allm=allm, w_all=w_all):
                        return proba(fit_w(make(params, s), X[allm], y[allm], w_all[allm]), X[te])
                    cands.append(dict(option=opt, algorithm=algo, features=fs, params=params,
                                      val_pred=proba(m, X[val]), refit=refit))
        finalize(part, algo, cands, meta)
        print(f"  {algo:20s} done ({(time.time() - t0) / 60:.0f} min)")
    if [a for a in NETS if a != "FT-Transformer"]:
        rounds_nets(part, {k: fsets[k] for k in NET_INPUTS}, y, gids, val, te, opts, meta)
    if not part.startswith("ranked"):
        report(part)
    report_research(part)


def run_ranked(part):
    """Ranked (app) model: live or round, WITHOUT Elo (the app cannot know how strong random ranked teams are)."""
    global TEST_IDS, VAL_IDS
    live = part == "ranked_live"
    target = "A_won" if live else "a_won_round"
    base = [f for f in (MAP_FEATURES if live else ROUND_FEATURES) if f not in ("elo_tuned", "elo_x_left")]
    print(f"\n================ RANKED {'LIVE (map winner before each round)' if live else 'ROUND (next-round winner)'}, no Elo")
    R = get_rounds()
    q = sorted(set(R["Game ID"]) & QUAL)
    VAL_IDS, TEST_IDS = set(q[int(len(q) * 0.6):int(len(q) * 0.8)]), set(q[int(len(q) * 0.8):])
    print(f"  qualifier maps with round data: {len(q):,}; training = earliest 60%, validation = next 20% "
          f"({len(VAL_IDS):,} maps), test = latest 20% ({len(TEST_IDS):,} maps)")
    R = R[np.isin(R["Game ID"], list(QUAL | OLD_PRO | PRO_2326))].reset_index(drop=True)
    gids = R["Game ID"].to_numpy()
    y = R[target].to_numpy().astype(float)
    E = state_features(R)[base]
    rich = E.copy()
    rich["score_a"], rich["score_b"] = R["sA"].astype(float), R["sB"].astype(float)
    rich["round_no"] = R["Round Number"].astype(float)
    rich["load_a_k"], rich["load_b_k"] = R["buy_a"].map(TYP).fillna(15000) / 1000, R["buy_b"].map(TYP).fillna(15000) / 1000
    rich["buy_a"], rich["buy_b"] = R["buy_a"].map(BUY).fillna(2), R["buy_b"].map(BUY).fillna(2)
    rich = rich.loc[:, ~rich.columns.duplicated()]
    raw = pd.DataFrame({"score_a": R["sA"].astype(float), "score_b": R["sB"].astype(float),
                        "round_no": R["Round Number"].astype(float), "a_attack": R["a_attack"].astype(float),
                        "buy_a": R["buy_a"].map(BUY).fillna(2), "buy_b": R["buy_b"].map(BUY).fillna(2)})
    fsets = {"engineered": E.to_numpy(), "rich": rich.to_numpy(), "raw": raw.to_numpy()}
    fit_all(part, R, y, gids, fsets, pick_opts(RANKED_OPTS))


def rounds_nets(part, Xsets, y, gids, val, te, opts, meta):
    """The rounds of each map in order; one prediction per round (only rounds already played are seen)."""
    starts = np.r_[0, np.flatnonzero(gids[1:] != gids[:-1]) + 1, len(gids)]
    T, nmap = int(np.diff(starts).max()), len(starts) - 1
    mg = gids[starts[:-1]]
    rowi = np.full((nmap, T), -1)
    mask = np.zeros((nmap, T), bool)
    for i in range(nmap):
        a, b = starts[i], starts[i + 1]
        rowi[i, :b - a], mask[i, :b - a] = np.arange(a, b), True
    ys = np.where(mask, y[np.maximum(rowi, 0)], 0).astype(np.float32)
    YS, MS = torch.tensor(ys), torch.tensor(mask)
    val_m, te_m = np.where(np.isin(mg, list(VAL_IDS)))[0], np.where(np.isin(mg, list(TEST_IDS)))[0]
    for arch in [a for a in NETS if a != "FT-Transformer"]:
        t0, cands = time.time(), []
        for (fsname, X), opt in [(xs, o) for xs in Xsets.items() for o in opts]:
            F = X.shape[1]
            w_fit, w_all = weights(gids, opt, False), weights(gids, opt, True)
            Xs = ((X - X[w_fit > 0].mean(0)) / (X[w_fit > 0].std(0) + 1e-6)).astype(np.float32)
            XS = torch.tensor(np.where(mask[..., None], Xs[np.maximum(rowi, 0)], 0).astype(np.float32))
            wf_map, wa_map = w_fit[starts[:-1]], w_all[starts[:-1]]
            fit_m, all_m = np.where(wf_map > 0)[0], np.where(wa_map > 0)[0]
            WF, WA = torch.tensor(wf_map, dtype=torch.float32), torch.tensor(wa_map, dtype=torch.float32)

            def preds(model, maps, XS=XS):
                out = np.zeros(len(gids))
                with torch.no_grad():
                    for i in range(0, len(maps), 512):
                        mm = maps[i:i + 512]
                        p = torch.sigmoid(model(XS[torch.as_tensor(mm, dtype=torch.long)])).numpy()
                        out[rowi[mm][mask[mm]]] = p[mask[mm]]
                return out

            for g in ParameterGrid(NET_GRID_SEQ):
                h, nl, dr, lr = int(g["hidden"]), int(g["layers"]), float(g["dropout"]), float(g["lr"])
                make = lambda h=h, nl=nl, dr=dr, F=F: SeqNet(arch, F, h, nl, dr, T + 1)

                def loss_fn(model, part_, ix, XS=XS, fit_m=fit_m, WF=WF):
                    mm = torch.as_tensor((fit_m if part_ == "fit" else val_m)[ix], dtype=torch.long)
                    l = nn.functional.binary_cross_entropy_with_logits(model(XS[mm]), YS[mm], reduction="none")
                    w = (WF[mm] if part_ == "fit" else torch.ones(len(mm)))[:, None] * MS[mm]
                    return (l * w).sum() / w.sum()
                model, ep = train_net(make, loss_fn, len(fit_m), len(val_m), lr, 0)
                vp = preds(model, val_m)[val]

                def refit(s, make=make, ep=ep, lr=lr, XS=XS, all_m=all_m, WA=WA, preds=preds):
                    def lf(model, part_, ix):
                        mm = torch.as_tensor(all_m[ix], dtype=torch.long)
                        l = nn.functional.binary_cross_entropy_with_logits(model(XS[mm]), YS[mm], reduction="none")
                        w = WA[mm][:, None] * MS[mm]
                        return (l * w).sum() / w.sum()
                    m, _ = train_net(make, lf, len(all_m), 0, lr, s, epochs=ep)
                    return np.clip(preds(m, te_m)[te], 1e-6, 1 - 1e-6)
                fs = f"{fsname}, {'each round' if arch == 'MLP' else 'in order'}"
                cands.append(dict(option=opt, algorithm=arch, features=fs, val_pred=np.clip(vp, 1e-6, 1 - 1e-6),
                                  params=dict(hidden=h, layers=nl, dropout=dr, lr=lr, epochs=ep), refit=refit))
        finalize(part, arch, cands, meta)
        print(f"  {arch:20s} done ({(time.time() - t0) / 60:.0f} min)")


start = time.time()
for part in PARTS:
    t0 = time.time()
    if RESEARCH_ONLY:
        report_research(part)
    elif REPORT_ONLY:
        report(part)
    elif part == "prematch":
        run_prematch()
    elif part.startswith("ranked"):
        run_ranked(part)
    else:
        run_rounds(part)
    print(f"  ({part} took {(time.time() - t0) / 60:.0f} min)")
print(f"\nDone in {(time.time() - start) / 60:.0f} min. Results in figures/final_selection_*{SFX}.csv. Send this output back.")
