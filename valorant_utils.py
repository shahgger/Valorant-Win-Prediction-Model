"""Shared helpers: agent roles and the per-map core/flex recommender."""
import pandas as pd

ROLES = {
    # Duelists
    "jett": "Duelist", "raze": "Duelist", "reyna": "Duelist", "phoenix": "Duelist",
    "yoru": "Duelist", "neon": "Duelist", "iso": "Duelist", "waylay": "Duelist",
    # Initiators
    "sova": "Initiator", "breach": "Initiator", "skye": "Initiator", "kayo": "Initiator",
    "fade": "Initiator", "gekko": "Initiator", "tejo": "Initiator",
    # Controllers
    "omen": "Controller", "brimstone": "Controller", "viper": "Controller",
    "astra": "Controller", "harbor": "Controller", "clove": "Controller", "miks": "Controller",
    # Sentinels
    "sage": "Sentinel", "cypher": "Sentinel", "killjoy": "Sentinel", "chamber": "Sentinel",
    "deadlock": "Sentinel", "vyse": "Sentinel", "veto": "Sentinel",
}


def recommend(data, map_name, years=(2025, 2026), k=20, top_n=5):
    """Returns (team_maps, core, flex, comps) for one map. Win rates exclude mirror picks."""
    rec = data[data["year"].isin(years) & (data["Map"] == map_name)]
    sides = pd.concat([
        rec.assign(comp=rec["comp_A"], opp=rec["comp_B"], won=rec["A_won"]),
        rec.assign(comp=rec["comp_B"], opp=rec["comp_A"], won=1 - rec["A_won"]),
    ])
    n = len(sides)

    # Agent level
    ex = sides.assign(agent=sides["comp"].str.split(",")).explode("agent")
    ex["mirror"] = [a in o.split(",") for a, o in zip(ex["agent"], ex["opp"])]
    ag = ex.groupby("agent").size().to_frame("picks")
    nm = ex[~ex["mirror"]].groupby("agent")["won"].agg(nm_games="size", nm_wins="sum")
    ag = ag.join(nm).fillna(0)
    ag["pick_rate"] = ag["picks"] / n
    ag["smoothed_wr"] = (ag["nm_wins"] + 0.5 * k) / (ag["nm_games"] + k)
    ag["role"] = ag.index.map(ROLES)
    cols = ["role", "picks", "pick_rate", "nm_games", "smoothed_wr"]
    core = ag[ag["pick_rate"] >= 0.6].sort_values("pick_rate", ascending=False)[cols]
    flex = ag[(ag["pick_rate"] >= 0.1) & (ag["pick_rate"] < 0.6)].sort_values("pick_rate", ascending=False)[cols]

    # Full compositions (exact mirror matchups excluded from the win rate)
    sides["mirror"] = sides["comp"] == sides["opp"]
    comps = sides.groupby("comp").size().to_frame("games")
    cnm = sides[~sides["mirror"]].groupby("comp")["won"].agg(nm_games="size", nm_wins="sum")
    comps = comps.join(cnm).fillna(0)
    comps["smoothed_wr"] = (comps["nm_wins"] + 0.5 * k) / (comps["nm_games"] + k)
    comps = comps[comps["games"] >= 10].sort_values("games", ascending=False).head(top_n)
    return n, core, flex, comps
