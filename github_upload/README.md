# Training-Data Weighting, Input Representation and Model Choice for Win Prediction in Professional VALORANT

Code, data and supplementary results for the paper submitted to NGES 2027. Every number in the paper's result tables
can be traced to a row in the appendix files, and the scripts below reproduce them from the raw data.

## Contents

| File or folder | Contents |
| --- | --- |
| `VALORANT_win_prediction_appendix.xlsx` | All result tables in one workbook, with a README sheet and a Summary sheet whose counts are formulas over the comparison sheets |
| `csv/` | The same tables as CSV files (see *Appendix tables* below) |
| `data/vct_2021` … `data/vct_2026` | The raw Kaggle dataset, in its original folder layout |
| `data/maps_dataset.csv` | One row per map: both teams' five-agent compositions and the result (built by `build_dataset.py`) |
| `data/maps_dataset_features.csv` | The same rows with every pre-match feature; the table all experiments read |
| `build_dataset.py` | Builds the two files above from the raw data |
| `features.py` | Elo ratings and the pre-match inputs (4 final and 11 match sets) |
| `round_model.py` | Round states and the live and round inputs (engineered, raw and rich sets), competitive levels |
| `valorant_utils.py` | Shared helpers (agent roles, loading) |
| `final_selection.py` | The main experiment: 15 training-data strategies × input sets × logistic regression, Extra Trees and Random Forest, grid search, the one-standard-error selection rule and the single test on pro 2026 |
| `paper_analysis.py` | The paired-bootstrap comparisons with Holm correction, calibration and the appendix tables |
| `data_inventory.py` | Data coverage and the noise ceiling (Table XI of the paper) |
| `list_dataset.py` | Lists every file, sheet and column of the raw dataset |
| `requirements.txt` | Python libraries |

## Data

The raw data is the Kaggle dataset *Valorant Champion Tour 2021–2026 Data* by Ryan Luong
(https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data), redistributed here under its
licence. Please cite the dataset as well as the paper if you use it.

Competitive levels used in the paper: **pro** = VCT 2023–2026 (franchised system); **pro-level** = 2021–2022 main
events, Masters, Champions and Last Chance Qualifiers; **qualifiers** = 2021–2022 open and closed qualifiers.

## Reproducing the results

Run everything from the repository root (the scripts read `data/...` by relative path and import each other).

```
python -m pip install -r requirements.txt
python build_dataset.py
python final_selection.py --part prematch --algos LR,ET,RF
python final_selection.py --part live --algos LR,ET,RF
python final_selection.py --part round --algos LR,ET,RF
python paper_analysis.py
python data_inventory.py
```

`build_dataset.py` can be skipped, because `data/maps_dataset_features.csv` is included. The three `final_selection.py`
runs fit every grid point for every strategy and input set (1,800 fits plus refits) and can take several hours; their
predictions are saved in `figures/final_selection_cache/`. `paper_analysis.py` then reads those predictions, trains
nothing, and writes the comparison tables to `figures/paper/`.

- **Splits:** test = pro 2026, used once (886 maps; 679 maps and 14,455 round states with round data). Validation =
  the latest 20% of pro 2023–2025 maps (642 pre-match; 431 live and round). Training = the rest of 2023–2025 plus the
  added data of each strategy.
- **Hyperparameters:** exhaustive grid search, scored by validation log loss. Logistic regression: C ∈ {0.001, 0.003,
  0.01, 0.03, 0.1, 0.3, 1}. Extra Trees and Random Forest: max depth ∈ {12, none} × min leaf ∈ {20, 100}, with 300
  trees (√p and 0.5p features per split). Tree finalists are refitted with three seeds.

## Appendix tables (`csv/`)

| File | Contents |
| --- | --- |
| `data.csv` | Maps, teams and round states per season and competitive level |
| `options.csv` | The 15 training-data strategies: weight of each season and level, and training-set size |
| `grid.csv` | Hyperparameter grids and fixed settings |
| `inputs.csv` | Input columns of each input representation |
| `finalists.csv` | All 360 finalists (task × strategy × algorithm × input set): chosen grid point, validation and test log loss, range over seeds, accuracy, Brier score, expected calibration error |
| `q1_data.csv` | RQ1: each strategy vs strategy B for the same algorithm and inputs (336 comparisons) |
| `q2_algorithms.csv` | RQ2: algorithm pairs on the same strategy and inputs (360 comparisons) |
| `q3_inputs.csv` | RQ3: input pairs on the same strategy and algorithm (225 comparisons) |
| `calibration_bins.csv` | Reliability bins of the strategy B models |

In the files, a training-data strategy is called an *option* (B, D, D½, C, Cf, E, Eq½, E½, Eo½, O, OQ, R¾, R½,
RQ¾, RQ½), as in the scripts.

**Comparison columns.** `log_loss_gain` = log loss of the reference minus that of the candidate on pro 2026 (positive
= candidate better). `ci95_low`, `ci95_high` = 95% percentile interval from a paired bootstrap that resamples whole test
maps (2,000 resamples). `p_value` = two-sided bootstrap p-value. `verdict_before_holm` = better or worse when the
interval excludes zero, same otherwise. `verdict_after_holm` = the same after Holm correction at 0.05 within each
family (Q1: one task × algorithm × input set; Q2: one task × input set × pair; Q3: one task × algorithm × input pair).
`seeds_agree` = in how many seeds the before-Holm verdict is the same.

## Citation

If you use this code or data, please cite the paper (citation to be added after publication) and the Kaggle dataset.
