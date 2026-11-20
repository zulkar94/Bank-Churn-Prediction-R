# Train and compare churn models on synthetic bank customers.
#
# Usage (from the repository root):
#   Rscript run_analysis.R

suppressPackageStartupMessages(library(glmnet))

source("R/data_generation.R")
source("R/metrics.R")
source("R/models.R")

dir.create("outputs", showWarnings = FALSE)

customers <- simulate_bank_customers(n = 5000, seed = 2026)
splits <- split_stratified(customers, train_fraction = 0.7, seed = 2026)
train <- splits$train
test <- splits$test

cat(sprintf("Train: %d customers (%.1f%% churned)\n", nrow(train), 100 * mean(train$churned)))
cat(sprintf("Test:  %d customers (%.1f%% churned)\n\n", nrow(test), 100 * mean(test$churned)))

logit_fit <- fit_logistic(train)
weighted_fit <- fit_logistic_weighted(train)
lasso_fit <- fit_lasso(train)

scores <- list(
  "Logistic regression" = stats::predict(logit_fit, newdata = test, type = "response"),
  "Logistic (class-weighted)" = stats::predict(weighted_fit, newdata = test, type = "response"),
  "Lasso logistic (CV)" = predict_lasso(lasso_fit, test)
)

comparison <- do.call(rbind, lapply(names(scores), function(name) {
  score <- scores[[name]]
  pr <- precision_recall(test$churned, score, threshold = 0.5)
  data.frame(
    model = name,
    auc = round(auc_rank(test$churned, score), 3),
    precision_at_0.5 = round(pr["precision"], 3),
    recall_at_0.5 = round(pr["recall"], 3),
    lift_top_20pct = round(lift_at(test$churned, score, 0.2), 2),
    row.names = NULL
  )
}))
write.csv(comparison, "outputs/model_comparison.csv", row.names = FALSE)
print(comparison)

# Coefficients from the plain logistic model: odds ratios with 95% intervals.
ci <- suppressMessages(stats::confint.default(logit_fit))
coefficients <- data.frame(
  term = names(stats::coef(logit_fit)),
  odds_ratio = round(exp(stats::coef(logit_fit)), 3),
  ci_low = round(exp(ci[, 1]), 3),
  ci_high = round(exp(ci[, 2]), 3),
  p_value = signif(summary(logit_fit)$coefficients[, 4], 3),
  row.names = NULL
)
# Balance is in pounds, so its per-£1 odds ratio rounds to 1.000. Report it per
# £10,000 instead, scaling the unrounded coefficient and interval.
is_balance <- coefficients$term == "balance"
coefficients$term[is_balance] <- "balance (per £10,000)"
coefficients$odds_ratio[is_balance] <- round(exp(10000 * stats::coef(logit_fit)[["balance"]]), 3)
coefficients$ci_low[is_balance] <- round(exp(10000 * ci["balance", 1]), 3)
coefficients$ci_high[is_balance] <- round(exp(10000 * ci["balance", 2]), 3)
write.csv(coefficients, "outputs/logistic_odds_ratios.csv", row.names = FALSE)
cat("\nOdds ratios (plain logistic model):\n")
print(coefficients)

# Gain table: share of churners captured when contacting the top x% of customers
# ranked by the score from the plain logistic model.
gain <- gain_table(test$churned, scores[["Logistic regression"]])
write.csv(gain, "outputs/gain_table.csv", row.names = FALSE)
print(gain)
