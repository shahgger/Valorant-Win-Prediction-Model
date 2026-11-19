"""
Round-by-round (in-game) win probability.

1. build_rounds(): one row per round of every map, from Team A's point of view: score before the round,
   which side Team A is on, both teams' loadout values and buy types, pre-match Elo, and the results.
   - Score: cumulative round wins before the round (eco_rounds outcomes).
   - Sides are not in the data, so they are inferred from how rounds ended: a team that wins by
     detonation (or loses a defused plant / fails to plant in time) was attacking; a team that wins
     by defusing or by time running out was defending. Each map's first-half side is a majority vote
     over those rounds; second half swaps; the overtime pattern is chosen by which fits the data better.
2. state_features(): the model inputs for a round state.
3. Two models (logistic regression):
   - map model:   P(Team A wins the map | state before the round)
   - round model: P(Team A wins this round | state)

Used by 05_extensions.ipynb (evaluation) and build_model.py / app.py (live spectator page).
"""
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

KEYS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]
BUY_TYPES = ["Eco", "Semi-eco", "Semi-buy", "Full buy"]
TYPICAL_LOADOUT = {"Eco": 3000, "Semi-eco": 8000, "Semi-buy": 15000, "Full buy": 22000}

MAP_FEATURES = ["elo_tuned", "lead", "na_inv", "nb_inv", "lead_x_left", "a_attack", "loadout_k",
                "a_eco", "b_eco", "elo_x_left"]
ROUND_FEATURES = ["elo_tuned", "a_attack", "loadout_k", "a_eco", "b_eco", "a_full", "b_full", "lead"]
# vlr.gg stopped showing loadout values from VCT Masters Toronto 2025 (empty for all of 2026), and app users usually
# know only the buy type. To train and test on the same kind of information, loadouts are estimated from the buy type
# everywhere (TYPICAL_LOADOUT) unless this is set to True.
USE_REAL_LOADOUT = False


def parse_k(v):
    """'3.9k' -> 3900; numbers pass through; anything else -> NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace(",", "")
    try:
        return float(s[:-1]) * 1000 if s.endswith("k") else float(s)
    except ValueError:
        return np.nan


def buy_label(t):
    t = str(t).lower()
    if t.startswith("full"):
        return "Full buy"
    if t.startswith("semi-buy") or t.startswith("semi buy"):
        return "Semi-buy"
    if t.startswith("semi-eco") or t.startswith("semi eco"):
        return "Semi-eco"
    if t.startswith("eco"):
        return "Eco"
    return np.nan


def side_from_method(method, outcome):
    """'att' / 'def' for the team in this row, or None if the method does not reveal the side."""
    m, won = str(method).lower(), str(outcome).lower() == "win"
    if won:
        if "detonat" in m:
            return "att"
        if "defus" in m or "time" in m:
            return "def"
    else:
        if "denied" in m or "failed to plant" in m:
            return "att"
        if "defus" in m:          # 'Defused Failed': the defenders failed to defuse
            return "def"
    return None


def first_half_phase(r, ot_same_as_first_on_odd=True):
    """True if round r has the same sides as round 1."""
    if r <= 12:
        return True
    if r <= 24:
        return False
    odd = (r % 2 == 1)
    return odd if ot_same_as_first_on_odd else not odd


def match_tier(tournament, stage):
    """Level of a 2021-2022 VCT match (before franchising, the Challengers main events were the top regional
    competition):  "open qualifier" (anyone could enter: the semi-pro / ranked-like level), "closed qualifier"
    (mixed: qualified tier-2 teams and invited top teams), "main event" (top regional teams), "international"
    (Masters / Champions), "lcq" (Last Chance Qualifiers).
    Ranked mode uses open + closed qualifiers: open qualifiers are the mixed / lower ranked-like level, closed
    qualifiers the tier-2 level that high-ranked (Immortal / Radiant) players reach."""
    t, s = str(tournament).lower(), str(stage).lower()
    if "masters" in t or "champions 20" in t or t.startswith("valorant champions"):
        return "international"
    if "last chance" in t:
        return "lcq"
    if "open qualifier" in s or "open qualifier" in t:
        return "open qualifier"
    if "qualifier" in s or "qualifier" in t:
        return "closed qualifier"
    return "main event"


SEMI_TIERS = {"open qualifier", "closed qualifier"}   # semi-pro / ranked = 2021-2022 open + closed qualifiers
PRO_TIERS = {"main event", "international", "lcq"}    # 2021-2022 pro-level matches


def tier_ids(data, years, tiers):
    """Game IDs of the given seasons whose match level is in `tiers`."""
    d = data[data["year"].isin(list(years))]
    keep = [match_tier(t, s) in tiers for t, s in zip(d["Tournament"], d["Stage"])]
    return set(d.loc[keep, "Game ID"])


SEMI_TEST_SHARE = 0.2                                 # semi-pro: latest 20% of the open-qualifier maps = test


def time_split(game_ids, test_share=SEMI_TEST_SHARE):
    """Split maps by time (Game ID order): the earliest (1 - test_share) for training, the latest for testing."""
    ids = sorted(set(game_ids))
    cut = int(round(len(ids) * (1 - test_share)))
    return set(ids[:cut]), set(ids[cut:])


def tournament_key(t):
    """Same event, different spellings: 'Champions Tour 2025: EMEA Stage 1' / 'VCT 2025: EMEA Stage 1' and
    'Champions Tour 2025: Masters Bangkok' / 'Valorant Masters Bangkok 2025' all become '2025 emea stage 1' etc."""
    t = str(t)
    yr = re.search(r"20\d\d", t)
    rest = re.sub(r"Champions Tour|VCT|Valorant|20\d\d|:", " ", t, flags=re.I)
    return f"{yr.group(0) if yr else ''} {' '.join(rest.lower().split())}"


def attach_maps(rows, maps, years, verbose=True):
    """Join round-file rows to the map table. First by the five keys; rows whose keys are spelled differently from
    the map table (e.g. 2025 eco_rounds.csv) are then matched by tournament + match name + map (tournament names
    compared after removing 'Champions Tour' / 'VCT' / 'Valorant'), and finally by match name + map within the same
    seasons, each only where that combination identifies a single map."""
    rows = rows.assign(_t=rows["Tournament"].map(tournament_key))
    maps = maps.assign(_t=maps["Tournament"].map(tournament_key))
    ALL = KEYS + ["_t"]
    exact = rows.merge(maps, on=ALL)
    done = set(map(tuple, exact[KEYS].drop_duplicates().to_numpy()))
    rest = rows[[tuple(k) not in done for k in rows[KEYS].to_numpy()]]
    out, n_fb = [exact], 0
    mp = maps[maps["year"].isin(list(years))]
    for keys in (["Tournament", "Match Name", "Map"], ["_t", "Match Name", "Map"], ["Match Name", "Map"]):
        if rest.empty:
            break
        other = [k for k in ALL if k not in keys]
        uniq = mp[~mp.duplicated(subset=keys, keep=False)].drop(columns=other)
        m = rest.drop(columns=other).merge(uniq, on=keys)
        if len(m):
            m = m.merge(maps[["Game ID"] + other], on="Game ID")
            out.append(m[exact.columns])
            n_fb += m[["Game ID"]].drop_duplicates().shape[0]
            hit = set(map(tuple, rest[keys].drop_duplicates().merge(uniq[keys], on=keys).to_numpy()))
            rest = rest[[tuple(k) not in hit for k in rest[keys].to_numpy()]]
    if verbose and n_fb:
        print(f"  matched {n_fb} more maps whose labels are spelled differently in the round files")
    return pd.concat(out, ignore_index=True).drop(columns="_t")


def build_rounds(data, years, verbose=True):
    """data: the map table (maps_dataset_features.csv). Returns one row per round, Team A's view."""
    eco = pd.concat([pd.read_csv(f"data/vct_{y}/matches/eco_rounds.csv", low_memory=False) for y in years])
    wl = pd.concat([pd.read_csv(f"data/vct_{y}/matches/win_loss_methods_round_number.csv", low_memory=False)
                    for y in years])
    maps = data[KEYS + ["Game ID", "year", "Team A", "Team B", "A_won", "elo_tuned"]].drop_duplicates(subset=KEYS, keep=False)

    eco = attach_maps(eco, maps, years, verbose)
    eco["loadout"] = eco["Loadout Value"].map(parse_k)
    eco["buy"] = eco["Type"].map(buy_label)
    eco["won_round"] = (eco["Outcome"].astype(str).str.lower() == "win").astype(int)
    a = eco[eco["Team"] == eco["Team A"]][["Game ID", "Round Number", "loadout", "buy", "won_round"]]
    b = eco[eco["Team"] == eco["Team B"]][["Game ID", "Round Number", "loadout", "buy"]]
    a = a.drop_duplicates(subset=["Game ID", "Round Number"], keep=False)
    b = b.drop_duplicates(subset=["Game ID", "Round Number"], keep=False)
    r = (a.rename(columns={"loadout": "la", "buy": "buy_a", "won_round": "a_won_round"})
          .merge(b.rename(columns={"loadout": "lb", "buy": "buy_b"}), on=["Game ID", "Round Number"])
          .merge(maps[["Game ID", "year", "Team A", "Team B", "A_won", "elo_tuned", "Map"]], on="Game ID"))
    r = r.sort_values(["Game ID", "Round Number"]).reset_index(drop=True)

    # complete maps only: rounds 1..N with no gaps
    g = r.groupby("Game ID")["Round Number"]
    complete = (g.transform("min") == 1) & (g.transform("max") == g.transform("size"))
    if verbose:
        print(f"Maps with round data: {r['Game ID'].nunique()} of {len(maps)}; "
              f"complete round sequences: {r.loc[complete, 'Game ID'].nunique()}")
    r = r[complete].copy()

    # score before each round
    r["sA"] = r.groupby("Game ID")["a_won_round"].cumsum() - r["a_won_round"]
    r["sB"] = (r["Round Number"] - 1) - r["sA"]

    # infer sides
    start, ot_pattern = infer_sides(wl, maps, verbose)
    r = r.merge(start, left_on="Game ID", right_index=True)
    same = r["Round Number"].map(lambda x: first_half_phase(x, ot_pattern))
    r["a_attack"] = np.where(same, r["a_att_first"], ~r["a_att_first"].astype(bool)).astype(int) * 2 - 1
    r["ot_pattern"] = ot_pattern
    return r.reset_index(drop=True)


def infer_sides(wl, maps, verbose=True):
    """Which side Team A starts on, per map, from how rounds ended (see module docstring).
    wl: win_loss_methods_round_number rows; maps: map table with KEYS, Game ID, Team A, Team B.
    Returns (Series a_att_first indexed by Game ID, overtime pattern)."""
    wl = wl.merge(maps[KEYS + ["Game ID", "Team A", "Team B"]], on=KEYS)
    wl["side"] = [side_from_method(m, o) for m, o in zip(wl["Method"], wl["Outcome"])]
    wl = wl.dropna(subset=["side"])
    wl = wl[(wl["Team"] == wl["Team A"]) | (wl["Team"] == wl["Team B"])].copy()
    wl["a_att"] = np.where(wl["Team"] == wl["Team A"], wl["side"] == "att", wl["side"] == "def")
    reg = wl[wl["Round Number"] <= 24].copy()
    reg["vote"] = np.where(reg["Round Number"] <= 12, reg["a_att"], ~reg["a_att"].astype(bool))
    votes = reg.groupby("Game ID")["vote"].agg(["mean", "size"])
    start = (votes["mean"] > 0.5).rename("a_att_first")
    agree = np.maximum(votes["mean"], 1 - votes["mean"])
    ot = wl[wl["Round Number"] > 24].merge(start, left_on="Game ID", right_index=True)
    fits = {}
    for pattern in (True, False):
        same = ot["Round Number"].map(lambda x: first_half_phase(x, pattern))
        pred = np.where(same, ot["a_att_first"], ~ot["a_att_first"].astype(bool))
        fits[pattern] = (pred == ot["a_att"].astype(bool)).mean() if len(ot) else np.nan
    ot_pattern = True if len(ot) == 0 or fits[True] >= fits[False] else False
    if verbose:
        print(f"Side inference: {len(votes)} maps with side evidence; evidence agreeing with the inferred "
              f"side: {(agree * votes['size']).sum() / votes['size'].sum():.1%} of informative rounds; "
              f"overtime pattern fit {max(fits.values()) if len(ot) else float('nan'):.1%}")
    return start, ot_pattern


def side_scores(data, years, verbose=True):
    """Attack and defence round wins per map from the round-by-round results and inferred sides
    (independent of the Attacker/Defender Score columns in maps_scores.csv).
    Returns one row per Game ID with the same column names the notebook uses for maps_scores."""
    wl = pd.concat([pd.read_csv(f"data/vct_{y}/matches/win_loss_methods_round_number.csv", low_memory=False)
                    for y in years])
    maps = data[KEYS + ["Game ID", "Team A", "Team B"]].drop_duplicates(subset=KEYS, keep=False)
    start, ot_pattern = infer_sides(wl, maps, verbose)
    w = wl[wl["Outcome"].astype(str).str.lower() == "win"].merge(maps, on=KEYS)
    w = w[(w["Team"] == w["Team A"]) | (w["Team"] == w["Team B"])]
    w = w.drop_duplicates(subset=["Game ID", "Round Number"], keep=False).merge(start, left_on="Game ID", right_index=True)
    same = w["Round Number"].map(lambda x: first_half_phase(x, ot_pattern))
    w["a_attacks"] = np.where(same, w["a_att_first"], ~w["a_att_first"].astype(bool))
    w["a_won"] = w["Team"] == w["Team A"]
    out = pd.DataFrame({
        "A Score": w.groupby("Game ID")["a_won"].sum(),
        "B Score": (~w["a_won"]).groupby(w["Game ID"]).sum(),
        "A Attacker Score": (w["a_won"] & w["a_attacks"]).groupby(w["Game ID"]).sum(),
        "A Defender Score": (w["a_won"] & ~w["a_attacks"]).groupby(w["Game ID"]).sum(),
        "B Attacker Score": (~w["a_won"] & ~w["a_attacks"]).groupby(w["Game ID"]).sum(),
        "B Defender Score": (~w["a_won"] & w["a_attacks"]).groupby(w["Game ID"]).sum(),
    }).astype(int).reset_index()
    return out


def state_features(df):
    """Model inputs from a round-state table with columns elo_tuned, sA, sB, a_attack, la, lb, buy_a, buy_b."""
    X = pd.DataFrame(index=df.index)
    sA, sB = df["sA"].astype(float), df["sB"].astype(float)
    ot = (sA >= 12) & (sB >= 12)
    na = np.where(ot, 2 - (sA - sB), 13 - sA).clip(1)       # rounds Team A still needs
    nb = np.where(ot, 2 - (sB - sA), 13 - sB).clip(1)
    left = (na + nb) / 26
    X["elo_tuned"] = df["elo_tuned"]
    X["lead"] = sA - sB
    X["na_inv"] = 1 / na
    X["nb_inv"] = 1 / nb
    X["lead_x_left"] = (sA - sB) / (na + nb)
    X["a_attack"] = df["a_attack"]
    if USE_REAL_LOADOUT:
        la = df["la"].fillna(df["buy_a"].map(TYPICAL_LOADOUT)).fillna(15000)
        lb = df["lb"].fillna(df["buy_b"].map(TYPICAL_LOADOUT)).fillna(15000)
    else:
        la = df["buy_a"].map(TYPICAL_LOADOUT).fillna(15000)
        lb = df["buy_b"].map(TYPICAL_LOADOUT).fillna(15000)
    X["loadout_k"] = (la - lb) / 1000
    X["a_eco"] = (df["buy_a"] == "Eco").astype(int)
    X["b_eco"] = (df["buy_b"] == "Eco").astype(int)
    X["a_full"] = (df["buy_a"] == "Full buy").astype(int)
    X["b_full"] = (df["buy_b"] == "Full buy").astype(int)
    X["elo_x_left"] = df["elo_tuned"] * left
    return X


def fit_models(rounds, sample_weight=None):
    X = state_features(rounds)
    map_m = LogisticRegression(C=1.0, max_iter=2000).fit(X[MAP_FEATURES], rounds["A_won"], sample_weight=sample_weight)
    round_m = LogisticRegression(C=1.0, max_iter=2000).fit(X[ROUND_FEATURES], rounds["a_won_round"],
                                                            sample_weight=sample_weight)
    return map_m, round_m


SEMI_ROUND_WEIGHT = 0.5      # semi-pro rounds in the ranked live model (test_live_semipro_training.py)


def live_models(live, ranked):
    """(map model, round model) for the app: ranked / student games use the model trained on pro + semi-pro rounds,
    team mode the pro-only model. Falls back to the pro-only model if round_state.pkl predates the ranked one."""
    if ranked and "map_m_ranked" in live:
        return live["map_m_ranked"], live["round_m_ranked"]
    return live["map_m"], live["round_m"]


def predict_state(map_m, round_m, elo_tuned, sA, sB, a_attack, buy_a, buy_b, la=None, lb=None):
    """Win probabilities for one live state (used by the app). a_attack: True if Team A attacks this round."""
    row = pd.DataFrame([{"elo_tuned": elo_tuned, "sA": sA, "sB": sB, "a_attack": 1 if a_attack else -1,
                         "la": la, "lb": lb, "buy_a": buy_a, "buy_b": buy_b}])
    X = state_features(row)
    return map_m.predict_proba(X[MAP_FEATURES])[0, 1], round_m.predict_proba(X[ROUND_FEATURES])[0, 1]
