# Bank Customer Churn Prediction in R

Churn models for a bank customer base, written in base R and `glmnet`. It simulates customers with known churn drivers, splits them into training and test sets, fits three models, and compares them on AUC, precision and recall, lift, and a gain table for targeting retention offers.

**The data is synthetic.** `R/data_generation.R` plants the churn effects on purpose, so the models can be checked for whether they recover them. Real bank data can replace the simulator if it has the same column names.

## What the analysis shows

On the 1,500-customer test set (seed 2026, 14.1% churn):

| Model | AUC | Precision at 0.5 | Recall at 0.5 | Lift, top 20% |
|---|---|---|---|---|
| Logistic regression | 0.761 | 0.486 | 0.085 | 2.32 |
| Logistic, class-weighted | 0.760 | 0.258 | 0.673 | 2.32 |
| Lasso logistic (CV) | 0.762 | 0.500 | 0.057 | 2.35 |

- **Ranking is the goal, not the 0.5 cut-off.** The three models rank customers almost identically (AUC 0.76). The class-weighted model trades precision for recall at 0.5, which only matters once a retention budget sets the threshold.
- **Contacting the top 20% captures 46% of churners.** The gain table in `outputs/gain_table.csv` shows the trade-off at each 5% step.
- **The planted effects come back.** In the logistic odds ratios, inactive membership (about 2.6x), a complaint (about 3.7x), Germany (about 2.3x), three or more products (about 2.0x) and tenure (about 0.92x per year) all point the expected way. Spain's effect is null, as planted.

## Method

- **Stratified split.** The 70/30 split keeps the churn rate the same in both sets.
- **Class weighting.** Inverse-frequency weights give churners the same total weight as non-churners. The model uses `quasibinomial` because binomial `glm` warns on non-integer weights. The coefficients are the same.
- **Lasso with CV on training data only.** `cv.glmnet` picks the penalty with 10-fold cross-validation on the training set. The test set is not used for tuning.
- **Metrics written from scratch.** AUC uses the rank (Mann-Whitney) formula, and the test suite checks it against hand-worked values. Lift is the top-group churn rate divided by the base rate.

## Quick start

Requires R 4.x with `glmnet` and `testthat`. On Ubuntu:

```bash
sudo apt-get install r-base-core r-cran-glmnet r-cran-testthat
```

```bash
Rscript run_analysis.R                # writes outputs/*.csv
Rscript -e 'testthat::test_dir("tests/testthat", stop_on_failure = TRUE)'
```

## Outputs

| File | Contents |
|---|---|
| `outputs/model_comparison.csv` | AUC, precision, recall and lift for each model |
| `outputs/logistic_odds_ratios.csv` | Odds ratios, 95% intervals and p-values |
| `outputs/gain_table.csv` | Share of churners captured by contacting the top x% |

Odds ratios for `balance` are per £1, so they read as 1.000. The odds ratio per £10,000 is the reported value raised to the power of 10,000.

## Limitations

- **Synthetic data.** The effects are planted, so the models are being checked against a known answer. They do not describe any real bank.
- **Base R models.** This version does not use tidymodels, since it was not available in the build environment. A tidymodels version with a proper resampling plan would be a natural next step.
- **No calibration check.** Predicted probabilities are not checked against observed churn rates by bin. This matters if the scores are used as probabilities rather than rankings.
- **Single split.** Results come from one train/test split and one seed. Repeated resampling would give confidence intervals on AUC.

## Project layout

```
R/data_generation.R     simulator and stratified split
R/metrics.R             AUC, precision/recall, lift
R/models.R              logistic, class-weighted and lasso fits
run_analysis.R          end-to-end comparison and outputs
tests/testthat/         unit tests for metrics, simulator and models
outputs/                generated tables
```

## Licence

MIT. See [LICENSE](LICENSE).
