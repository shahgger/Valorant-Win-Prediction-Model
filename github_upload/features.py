"""
Feature engineering for the Valorant FYP.

Every feature uses only games played BEFORE the map it describes: maps are processed in
Game ID order (the dataset's match order), each feature is read from the history so far,
and only then is the history updated with the map's result. All features are expressed
as a difference, Team A minus Team B.

Used by 02_modeling.ipynb:   data = build_all_features(pd.read_csv("data/maps_dataset.csv"))
"""
from collections import deque

import numpy as np
import pandas as pd

KEYS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]
YEARS = range(2021, 2027)


# ---------------------------------------------------------------- team strength (Elo)
def compute_elo(data, K, reg):
    """Elo difference (A - B) / 400 before each map.
    K = update size; reg = share pulled back toward 1500 at the start of each season."""
    elo, out, cur_year = {}, [], None
    for ta, tb, won, yr in zip(data["Team A"], data["Team B"], data["A_won"], data["year"]):
        if yr != cur_year:
            elo = {t: 1500 + (1 - reg) * (e - 1500) for t, e in elo.items()}
            cur_year = yr
        ra, rb = elo.get(ta, 1500), elo.get(tb, 1500)
        out.append((ra - rb) / 400)
        exp_a = 1 / (1 + 10 ** ((rb - ra) / 400))
        elo[ta] = ra + K * (won - exp_a)
        elo[tb] = rb - K * (won - exp_a)
    return np.array(out)


# ---------------------------------------------------------------- composition features
def add_composition_features(data, k=50, k2=10):
    """comp_diff: mean smoothed win rate of the five agents on this map (prior weight k).
    fam_diff: log(1 + times the team has played this exact comp on this map).
    teammap_diff: the team's smoothed win rate on this map (prior weight k2)."""
    agent_hist, team_map, team_comp = {}, {}, {}
    comp_d, fam_d, tm_d = [], [], []

    def strength(comp, m):
        vals = []
        for a in comp.split(","):
            w, n = agent_hist.get((m, a), (0, 0))
            vals.append((w + 0.5 * k) / (n + k))
        return np.mean(vals)

    def team_map_wr(team, m):
        w, n = team_map.get((team, m), (0, 0))
        return (w + 0.5 * k2) / (n + k2)

    for m, ta, tb, ca, cb, won in zip(data["Map"], data["Team A"], data["Team B"],
                                      data["comp_A"], data["comp_B"], data["A_won"]):
        comp_d.append((strength(ca, m) - strength(cb, m)) * 100)
        fam_d.append(np.log1p(team_comp.get((ta, m, ca), 0)) - np.log1p(team_comp.get((tb, m, cb), 0)))
        tm_d.append((team_map_wr(ta, m) - team_map_wr(tb, m)) * 10)
        for team, comp, w_ in [(ta, ca, won), (tb, cb, 1 - won)]:
            for a in comp.split(","):
                w0, n0 = agent_hist.get((m, a), (0, 0))
                agent_hist[(m, a)] = (w0 + w_, n0 + 1)
            team_comp[(team, m, comp)] = team_comp.get((team, m, comp), 0) + 1
            w0, n0 = team_map.get((team, m), (0, 0))
            team_map[(team, m)] = (w0 + w_, n0 + 1)

    data["comp_diff"], data["fam_diff"], data["teammap_diff"] = comp_d, fam_d, tm_d
    return data


# ---------------------------------------------------------------- player rows
def load_player_rows(data, years=YEARS, verbose=True):
    """One row per player per map (full-map totals), linked to Game ID, with a skill score:
    vlr.gg Rating, or a Rating estimated from ACS where Rating is missing."""
    ov_list, id_list = [], []
    for yr in years:
        ov = pd.read_csv(f"data/vct_{yr}/matches/overview.csv", low_memory=False)
        ov = ov[(ov["Side"] == "both") & (ov["Map"] != "All Maps")]
        ov_list.append(ov[KEYS + ["Team", "Player", "Agents", "Rating", "Average Combat Score"]])
        ids = pd.read_csv(f"data/vct_{yr}/ids/tournaments_stages_matches_games_ids.csv", low_memory=False)
        id_list.append(ids[KEYS + ["Game ID"]])

    ids = pd.concat(id_list).drop_duplicates(subset=KEYS, keep="first")
    ov = pd.concat(ov_list).merge(ids, on=KEYS)
    ov["Rating"] = pd.to_numeric(ov["Rating"], errors="coerce")
    ov["ACS"] = pd.to_numeric(ov["Average Combat Score"], errors="coerce")
    both = ov.dropna(subset=["Rating", "ACS"])
    a, b = np.polyfit(both["ACS"], both["Rating"], 1)
    ov["skill_score"] = ov["Rating"].fillna(a * ov["ACS"] + b)
    ov = ov[ov["Game ID"].isin(data["Game ID"])]
    if verbose:
        r = both["Rating"].corr(both["ACS"])
        print(f"Rating = {a:.4f} * ACS + {b:.3f}  (correlation {r:.2f}); "
              f"{ov['Rating'].isna().sum()} of {len(ov)} player rows use the ACS estimate")
    return ov


# ---------------------------------------------------------------- player skill and spread
def add_player_features(data, ov, alpha=0.1, kp=5, cap=20):
    """player_diff: mean rolling skill of each lineup (recency-weighted Rating, shrunk to 1.0
    for players with few maps). spread_diff: standard deviation of the lineup's skill."""
    sk = ov.dropna(subset=["skill_score"])
    game_players = {}
    for g, t, p, s in zip(sk["Game ID"], sk["Team"], sk["Player"], sk["skill_score"]):
        game_players.setdefault((g, t), []).append((p, s))

    phist = {}

    def player_skill(p):
        if p not in phist:
            return 1.0
        e, n = phist[p]
        n = min(n, cap)
        return (e * n + 1.0 * kp) / (n + kp)

    player_d, spread_d, covered = [], [], 0
    for g, ta, tb in zip(data["Game ID"], data["Team A"], data["Team B"]):
        pa, pb = game_players.get((g, ta), []), game_players.get((g, tb), [])
        covered += bool(pa and pb)
        va, vb = [player_skill(p) for p, _ in pa], [player_skill(p) for p, _ in pb]
        player_d.append(((np.mean(va) if pa else 1.0) - (np.mean(vb) if pb else 1.0)) * 10)
        spread_d.append(((np.std(va) if len(pa) > 1 else 0.0) - (np.std(vb) if len(pb) > 1 else 0.0)) * 10)
        for p, rating in pa + pb:
            e, n = phist.get(p, (rating, 0))
            phist[p] = ((1 - alpha) * e + alpha * rating if n else rating, n + 1)

    data["player_diff"], data["spread_diff"] = player_d, spread_d
    print(f"Maps with both rosters found: {covered}/{len(data)}")
    return data


# ---------------------------------------------------------------- map pick
def add_map_pick(data, years=YEARS):
    """+1 if Team A picked the map, -1 if Team B did, 0 for deciders or unknown."""
    dp = pd.concat([pd.read_csv(f"data/vct_{yr}/matches/draft_phase.csv") for yr in years])
    picks = (dp[dp["Action"].str.lower() == "pick"][KEYS + ["Team"]]
             .drop_duplicates(subset=KEYS, keep=False)
             .rename(columns={"Team": "picker"}))
    data = data.merge(picks, on=KEYS, how="left")
    data["map_pick"] = np.where(data["picker"] == data["Team A"], 1,
                                np.where(data["picker"] == data["Team B"], -1, 0))
    return data.drop(columns=["picker"])


# ---------------------------------------------------------------- form and head-to-head
def add_form_h2h(data, last_n=10):
    """form_diff: win rate over each team's last N maps. h2h_diff: smoothed record vs this opponent."""
    form, h2h, form_d, h2h_d = {}, {}, [], []
    for ta, tb, won in zip(data["Team A"], data["Team B"], data["A_won"]):
        fa, fb = form.setdefault(ta, deque(maxlen=last_n)), form.setdefault(tb, deque(maxlen=last_n))
        form_d.append(((sum(fa) + 1) / (len(fa) + 2) - (sum(fb) + 1) / (len(fb) + 2)) * 10)
        key = tuple(sorted((ta, tb)))
        w, n = h2h.get(key, (0, 0))                     # w = wins of key[0]
        wa = w if key[0] == ta else n - w
        h2h_d.append(((wa + 1) / (n + 2) - 0.5) * 10)
        fa.append(won)
        fb.append(1 - won)
        h2h[key] = (w + (won if key[0] == ta else 1 - won), n + 1)
    data["form_diff"], data["h2h_diff"] = form_d, h2h_d
    return data


# ---------------------------------------------------------------- event importance
def add_event_interactions(data, elo_col="elo_tuned"):
    """Elo difference multiplied by international-event and playoff indicators."""
    intl = data["Tournament"].str.contains("Masters|Valorant Champions 20", regex=True)
    playoff = (data["Stage"].str.contains("Playoff", case=False, na=False)
               | data["Match Type"].str.contains("Final", case=False, na=False))
    data["elo_x_intl"] = data[elo_col] * intl
    data["elo_x_playoff"] = data[elo_col] * playoff
    return data


# ---------------------------------------------------------------- player-agent comfort
def add_comfort(data, ov):
    """comfort_diff: mean log(1 + times each player has played their current agent before)."""
    pl = ov.dropna(subset=["Agents"]).copy()
    pl["Agents"] = pl["Agents"].str.lower().str.strip()
    gp = {}
    for g, t, p, a in zip(pl["Game ID"], pl["Team"], pl["Player"], pl["Agents"]):
        gp.setdefault((g, t), []).append((p, a))

    pa_hist, comfort_d = {}, []
    for g, ta, tb in zip(data["Game ID"], data["Team A"], data["Team B"]):
        vals = []
        for t in (ta, tb):
            players = gp.get((g, t), [])
            vals.append(np.mean([np.log1p(pa_hist.get(x, 0)) for x in players]) if players else 0.0)
        comfort_d.append(vals[0] - vals[1])
        for t in (ta, tb):
            for x in gp.get((g, t), []):
                pa_hist[x] = pa_hist.get(x, 0) + 1
    data["comfort_diff"] = comfort_d
    return data


# ---------------------------------------------------------------- features added to raise accuracy (02, section 8b)
def compute_elo_mov(data, K=16, reg=0.25):
    """Elo that also uses the score margin: a 13-2 win moves ratings more than 13-11.
    Update = K x log(1 + round difference) x (won - expected), with the usual correction that shrinks the update
    when the favourite wins (so strong teams' ratings do not run away). Returns (A - B) / 400 before each map."""
    elo, out, cur_year = {}, [], None
    for ta, tb, sa, sb, won, yr in zip(data["Team A"], data["Team B"], data["Team A Score"], data["Team B Score"],
                                       data["A_won"], data["year"]):
        if yr != cur_year:
            elo = {t: 1500 + (1 - reg) * (e - 1500) for t, e in elo.items()}
            cur_year = yr
        ra, rb = elo.get(ta, 1500), elo.get(tb, 1500)
        out.append((ra - rb) / 400)
        exp_a = 1 / (1 + 10 ** ((rb - ra) / 400))
        winner_edge = (ra - rb) if won else (rb - ra)
        mult = np.log(abs(float(sa) - float(sb)) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)
        elo[ta] = ra + K * mult * (won - exp_a)
        elo[tb] = rb - K * mult * (won - exp_a)
    return np.array(out)


def compute_map_elo(data, K=16, reg=0.25, Km=8, map_reg=0.5):
    """Map-specific strength: the team's overall Elo plus a per-map offset that learns from results on that map.
    Km = offset update size; at each new season every offset shrinks by map_reg (0.5 = halved).
    Returns ((A overall + A map offset) - (B overall + B map offset)) / 400 before each map."""
    elo, off, out, cur_year = {}, {}, [], None
    for ta, tb, m, won, yr in zip(data["Team A"], data["Team B"], data["Map"], data["A_won"], data["year"]):
        if yr != cur_year:
            elo = {t: 1500 + (1 - reg) * (e - 1500) for t, e in elo.items()}
            off = {k: (1 - map_reg) * v for k, v in off.items()}
            cur_year = yr
        ra, rb = elo.get(ta, 1500), elo.get(tb, 1500)
        oa, ob = off.get((ta, m), 0.0), off.get((tb, m), 0.0)
        out.append(((ra + oa) - (rb + ob)) / 400)
        exp_base = 1 / (1 + 10 ** ((rb - ra) / 400))
        exp_map = 1 / (1 + 10 ** (((rb + ob) - (ra + oa)) / 400))
        elo[ta] = ra + K * (won - exp_base)
        elo[tb] = rb - K * (won - exp_base)
        off[(ta, m)] = oa + Km * (won - exp_map)
        off[(tb, m)] = ob - Km * (won - exp_map)
    return np.array(out)


def add_series_context(data, elo_col="elo_tuned"):
    """series_lead: maps Team A has won minus maps Team B has won earlier in the same series (known before this map).
    elo_x_mapno: Elo difference x (map number - 1), letting team strength matter more or less later in a series."""
    series = list(map(tuple, data[["Tournament", "Stage", "Match Type", "Match Name"]].to_numpy()))
    wins, lead, mapno = {}, [], []
    for key, ta, tb, won in zip(series, data["Team A"], data["Team B"], data["A_won"]):
        w = wins.setdefault(key, {})
        lead.append(w.get(ta, 0) - w.get(tb, 0))
        mapno.append(sum(w.values()) + 1)
        winner = ta if won else tb
        w[winner] = w.get(winner, 0) + 1
    data["series_lead"] = lead
    data["map_number"] = mapno
    data["elo_x_mapno"] = data[elo_col] * (data["map_number"] - 1)
    return data


def add_roster_change(data, ov, window=10):
    """roster_new_diff: number of players in Team A's lineup who did not play for the team in its previous
    `window` maps, minus the same for Team B (0 when a lineup or the team's history is unknown)."""
    gp = {}
    for g, t, pl in zip(ov["Game ID"], ov["Team"], ov["Player"]):
        gp.setdefault((g, t), set()).add(pl)
    hist, out = {}, []
    for g, ta, tb in zip(data["Game ID"], data["Team A"], data["Team B"]):
        vals = []
        for t in (ta, tb):
            cur, past = gp.get((g, t)), hist.get(t)
            vals.append(len(cur - set().union(*past)) if cur and past else 0)
        out.append(vals[0] - vals[1])
        for t in (ta, tb):
            if gp.get((g, t)):
                hist.setdefault(t, deque(maxlen=window)).append(gp[(g, t)])
    data["roster_new_diff"] = out
    return data


# ---------------------------------------------------------------- everything
def build_all_features(data):
    """Adds every feature to the map table, in match order."""
    data = data.sort_values("Game ID").reset_index(drop=True)
    data["elo_diff"] = compute_elo(data, 32, 0)        # default Elo
    data["elo_tuned"] = compute_elo(data, 16, 0.25)    # tuned Elo (final model)
    data = add_composition_features(data)
    ov = load_player_rows(data)
    data = add_player_features(data, ov)
    data = add_map_pick(data)
    data = data.sort_values("Game ID").reset_index(drop=True)
    data = add_form_h2h(data)
    data = add_event_interactions(data)
    data = add_comfort(data, ov)
    data["elo_mov"] = compute_elo_mov(data, 16, 0.25)
    data["elo_map"] = compute_map_elo(data, 16, 0.25, 8, 0.5)
    data = add_series_context(data)
    data = add_roster_change(data, ov)
    return data
