# HepatoIA model card

## Version

`hepatoia-severity3-softmax-v0.1`

## Intended use

Clinical decision support for non-invasive severity screening of hepatic steatosis. The model estimates calibrated probabilities across 3 severity tiers and must be interpreted by a clinician. It is not an autonomous diagnostic system.

## Data

Two combined NHANES cycles, unified by [build_hepatoia_dataset.py](build_hepatoia_dataset.py):

| Cycle | Raw files | SDDSRVYR code | N after filters |
|---|---|---|---:|
| 2017-March 2020 (pre-pandemic pooled release) | `data/raw/nhanes_2017_2020/P_*.xpt` | 66 | 6,823 |
| August 2021-2023 | `data/raw/nhanes_2021_2023/*_L.xpt` | 12 | 4,682 |
| **Combined** | | | **11,505** |

Note: the `nhanes_2017_2020/P_*.xpt` files are the **2017-March 2020** pre-pandemic pooled cycle (identified by the `WTINTPRP`/`WTMECPRP` sampling-weight variables), not "August 2021-2023" as earlier versions of this card stated. The `nhanes_2021_2023/*_L.xpt` files are the actual August 2021-2023 standalone cycle (`WTINT2YR`/`WTMEC2YR`, cycle letter "L"). The two cycles cover disjoint `SEQN` ranges (109263-124822 vs. 130378-142310) — confirmed non-overlapping populations before merging.

Cohort filters applied to each cycle before combining:
- Adults (age ≥ 18)
- Not pregnant
- Complete VCTE/FibroScan exam (`LUAXSTAT == 1`)
- Reliable LSM reading (IQR/median < 30%)
- **Excluded: excessive alcohol use** (avg ≥15 drinks/day, or drinking ≥3-4 times/week — same definition as Lin et al. 2025, pone.0319851), so the label reflects non-alcoholic steatosis specifically.

Output dataset: `artifacts/hepatoia_final_dataset_v2.csv`

Class balance (3-tier severity, combined cohort):

- Normal: 5,057
- Leve_Moderada: 2,146
- Severa: 4,302

## Target

3-tier severity classification, from CAP-derived grade (`Steatosis_Grade_CAP`):

- `Normal` — `S0_Normal`
- `Leve_Moderada` — `S1_Leve` or `S2_Moderada` (merged to avoid a severely underrepresented middle class)
- `Severa` — `S3_Severa`

## Predictors

- `Waist_cm`
- `TG_HDL_ratio`
- `GGT`
- `Uric_Acid`
- `AST_ALT_ratio`
- `TyG_index`
- `Age`
- `HDL_BMI_ratio`
- `NHHR` (non-HDL-C / HDL-C — added from Lin et al. 2025, pone.0319851)

Blocked from predictors because of leakage or identity:

- `CAP_dBm`
- `LSM_kPa`
- `Steatosis_Grade_CAP`
- `Fibrosis_Grade_LSM`
- `SEQN`

## Model

Regularized multinomial logistic regression (softmax), implemented from scratch in `numpy` — see [train_hepatoia_model.py](train_hepatoia_model.py). Class-weighted gradient descent, L2 regularization, temperature-scaling calibration (multiclass generalization of Platt scaling). Decision rule is argmax over calibrated class probabilities.

## Test performance (combined cohort, 3-tier severity)

| Metric (macro) | Value |
|---|---:|
| Accuracy | 0.6335 |
| AUC (macro, one-vs-rest) | 0.7916 |
| Precision (macro) | 0.5878 |
| Sensitivity/recall (macro) | 0.5845 |
| Specificity (macro) | 0.8198 |
| F1 (macro) | 0.5839 |
| Brier score (multiclass) | 0.4790 |
| Accuracy within ±1 adjacent class | 0.9187 |
| Severe misclassification (Normal↔Severa confusion) | 0.0813 |

Per-class AUC:

| Class | AUC | F1 |
|---|---:|---:|
| Normal | 0.8536 | 0.7304 |
| Leve_Moderada | 0.6654 | 0.3195 |
| Severa | 0.8557 | 0.7017 |

Confusion matrix on test (rows = actual, columns = predicted):

| | Pred Normal | Pred Leve_Moderada | Pred Severa |
|---|---:|---:|---:|
| Actual Normal | 695 | 197 | 119 |
| Actual Leve_Moderada | 129 | 154 | 146 |
| Actual Severa | 68 | 184 | 608 |

The middle class (`Leve_Moderada`) is consistently the weakest across every published NHANES-based study we compared against (e.g. Wang et al. 2025, BMC Gastroenterology, PMC11998142, reports AUC 0.66 for their moderate-risk tier using Random Forest) — the "±1 adjacent class" accuracy (91.9%) and low severe-misclassification rate (8.1%) are the more clinically meaningful numbers: the model rarely confuses the two extremes.

## Strongest coefficients (by class, absolute standardized weight)

See `coefficient_table` in `artifacts/hepatoia_model.json` for the full per-class breakdown — each feature now has 3 coefficients (one per class) instead of 1.

## Known limitations

- Internal model, not externally validated.
- NHANES sampling weights are not yet incorporated (unweighted pooled analysis).
- `Leve_Moderada` class has meaningfully weaker discrimination than the two extreme classes — likely a ceiling effect from CAP being a continuous measurement force-cut into discrete tiers, not solely a data or model deficiency (same pattern reported in comparable NHANES-based studies).
- Calibration should be inspected visually before clinical presentation.
- Subgroup performance should be checked by sex, age band, and other clinically relevant groups.
- No k-fold cross-validation yet — single stratified train/calibration/test split.
- A medical-device style validation plan is still needed before any real clinical use.
