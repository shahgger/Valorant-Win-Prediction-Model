# Appendix: Does Older Data Help? Training-Data Weighting, Input Representation and Model Choice for Win Prediction in Professional VALORANT

Supplementary data for the paper submitted to NGES 2027. Every number in the paper's result tables can be traced to a
row in these files.

## Files

| File | Contents |
| --- | --- |
| `VALORANT_win_prediction_appendix.xlsx` | All tables below in one workbook, with a README sheet and a Summary sheet whose counts are formulas over the comparison sheets |
| `csv/data.csv` | Maps, teams and round states per season and competitive level |
| `csv/options.csv` | The 15 training-data options: weight of each season and level, and training-set size |
| `csv/grid.csv` | Hyperparameter grids (exhaustive grid search) and fixed settings |
| `csv/inputs.csv` | Input columns of each input representation |
| `csv/finalists.csv` | All 360 finalists (task × option × algorithm × input set): chosen grid point, validation and test log loss, range over seeds, accuracy, Brier score, expected calibration error |
| `csv/q1_data.csv` | RQ1: each option vs option B for the same algorithm and inputs (336 comparisons) |
| `csv/q2_algorithms.csv` | RQ2: algorithm pairs on the same option and inputs (360 comparisons) |
| `csv/q3_inputs.csv` | RQ3: input pairs on the same option and algorithm (225 comparisons) |
| `csv/calibration_bins.csv` | Reliability bins of the option B models |
| `paper_analysis.py` | The script that computed the comparisons from the saved model predictions |

## Set-up in brief

- **Data:** Kaggle dataset *Valorant Champion Tour 2021–2026 Data* (ryanluong1). Pro = VCT 2023–2026;
  pro-level = 2021–22 main events, Masters, Champions and Last Chance Qualifiers; qualifiers = 2021–22 open and closed
  qualifiers.
- **Tasks:** pre-match (map winner before the map), live (map winner before each round), round (next-round winner).
- **Test:** pro 2026, used once: 886 maps; 679 maps and 14,455 round states with round data.
- **Validation:** the latest 20% of pro 2023–2025 maps (642 pre-match; 431 live/round).
- **Algorithms:** logistic regression, Extra Trees and Random Forest (scikit-learn). Each grid point was fitted for
  every option and input set and scored by validation log loss. The best per option × algorithm × input set is a
  finalist, which is refitted on training plus validation data and tested. Trees were refitted with 3 seeds.

## Comparison columns

- `log_loss_gain`: mean log loss of the reference minus that of the candidate on pro 2026 (positive = candidate
  better).
- `ci95_low`, `ci95_high`: 95% percentile interval from a paired bootstrap that resamples whole test maps (2,000
  resamples).
- `p_value`: two-sided bootstrap p-value.
- `verdict_before_holm`: `better` or `worse` when the interval excludes zero, `same` otherwise.
- `verdict_after_holm`: the same verdict after Holm correction at 0.05 within each family:
  - Q1: one task × algorithm × input set;
  - Q2: one task × input set × pair;
  - Q3: one task × algorithm × input pair.
- `seeds_agree`: in how many of the seeds the before-Holm verdict is the same (logistic regression has one).
