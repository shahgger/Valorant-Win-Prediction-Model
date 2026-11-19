"""
Builds the two dataset files every experiment reads, from the raw Kaggle folders data/vct_2021 ... data/vct_2026:

    data/maps_dataset.csv            one row per map: both teams' five-agent compositions and the result
    data/maps_dataset_features.csv   the same rows plus every pre-match feature (Elo, form, familiarity, ...)

    python build_dataset.py

This is the data-preparation part of notebooks 01 and 02 as one script. Team A and Team B are swapped at random
(seed 42) so the listing order carries no information about the result.
"""
import numpy as np
import pandas as pd

from features import build_all_features

YEARS = range(2021, 2027)
KEYS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]


def load_year(y):
    base = f"data/vct_{y}"
    ov = pd.read_csv(f"{base}/matches/overview.csv", low_memory=False)
    ms = pd.read_csv(f"{base}/matches/maps_scores.csv", low_memory=False)
    ids = pd.read_csv(f"{base}/ids/tournaments_stages_matches_games_ids.csv", low_memory=False)

    # one row per player per map (drop attack/defend splits and series totals)
    ov = ov[(ov["Side"] == "both") & (ov["Map"] != "All Maps")]
    ov = ov.dropna(subset=["Agents", "Team"]).drop_duplicates(subset=KEYS + ["Team", "Player"])

    # each team's five agents on each map, sorted so the order does not matter
    comps = (ov.groupby(KEYS + ["Team"])["Agents"]
               .apply(lambda s: sorted(s.str.lower().str.strip()))
               .reset_index(name="comp"))
    comps = comps[comps["comp"].str.len() == 5]
    comps["comp"] = comps["comp"].str.join(",")

    # attach both teams' compositions to the map result
    ms = ms[KEYS + ["Team A", "Team B", "Team A Score", "Team B Score"]]
    a = comps.rename(columns={"Team": "Team A", "comp": "comp_A"})
    b = comps.rename(columns={"Team": "Team B", "comp": "comp_B"})
    df = ms.merge(a, on=KEYS + ["Team A"]).merge(b, on=KEYS + ["Team B"])

    # Game ID increases with time and serves as the match order (the dataset has no dates)
    ids = ids[KEYS + ["Match ID", "Game ID"]].drop_duplicates(subset=KEYS)
    df = df.merge(ids, on=KEYS, how="left")
    df["year"] = y
    print(f"{y}: {len(ms)} maps in scores -> {len(df)} matched with compositions")
    return df


data = pd.concat([load_year(y) for y in YEARS], ignore_index=True)
data = data.sort_values("Game ID").reset_index(drop=True)

# random Team A / Team B order, label, and removal of maps sharing a Game ID
rng = np.random.default_rng(42)
swap = rng.random(len(data)) < 0.5
for c1, c2 in [("Team A", "Team B"), ("comp_A", "comp_B"), ("Team A Score", "Team B Score")]:
    data.loc[swap, [c1, c2]] = data.loc[swap, [c2, c1]].to_numpy()
data["A_won"] = (data["Team A Score"] > data["Team B Score"]).astype(int)
data = data[~data["Game ID"].duplicated(keep=False) & data["Game ID"].notna()]
data.to_csv("data/maps_dataset.csv", index=False)
print("clean maps:", len(data))

features = build_all_features(pd.read_csv("data/maps_dataset.csv"))
features.to_csv("data/maps_dataset_features.csv", index=False)
print("saved data/maps_dataset.csv and data/maps_dataset_features.csv", features.shape)
