# Tests for the churn helpers. Run from the repository root:
#   Rscript -e 'testthat::test_dir("tests/testthat", stop_on_failure = TRUE)'

library(glmnet)
source(testthat::test_path("../../R/data_generation.R"))
source(testthat::test_path("../../R/metrics.R"))
source(testthat::test_path("../../R/models.R"))

test_that("AUC matches hand-worked values", {
  # Perfect separation gives 1, reversed gives 0, and a tie gives 0.5.
  expect_equal(auc_rank(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9)), 1)
  expect_equal(auc_rank(c(0, 0, 1, 1), c(0.9, 0.8, 0.2, 0.1)), 0)
  expect_equal(auc_rank(c(0, 1), c(0.5, 0.5)), 0.5)
  # Positives at 0.6 and 0.2, negatives at 0.4 and 0.1: 3 of 4 pairs ranked correctly.
  expect_equal(auc_rank(c(0, 0, 1, 1), c(0.1, 0.4, 0.2, 0.6)), 0.75)
})

test_that("AUC refuses single-class input", {
  expect_error(auc_rank(c(1, 1, 1), c(0.2, 0.5, 0.9)))
})

test_that("precision and recall at a threshold", {
  actual <- c(1, 1, 0, 0, 0)
  score <- c(0.9, 0.3, 0.6, 0.1, 0.2)
  pr <- precision_recall(actual, score, threshold = 0.5)
  expect_equal(unname(pr["precision"]), 0.5)  # 1 true positive of 2 predicted
  expect_equal(unname(pr["recall"]), 0.5)     # 1 of 2 churners found
})

test_that("lift in the top 20% is computed from the ranking", {
  actual <- c(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)  # base rate 0.2
  score <- seq(10, 1)                         # the two churners are ranked first
  expect_equal(lift_at(actual, score, 0.2), 5) # top 20% (2 customers) are all churners
})

test_that("lift and gain ignore row order when scores are tied", {
  # A score with no information must give lift 1, wherever the churners sit.
  actual <- c(rep(1, 10), rep(0, 90))
  flat <- rep(0.5, 100)
  expect_equal(lift_at(actual, flat, 0.2), 1)
  expect_equal(lift_at(rev(actual), flat, 0.2), 1)
  expect_equal(gain_table(actual, flat, c(20, 50))$churners_captured_pct, c(20, 50))
  # Partial tie at the cut-off: one clear churner on top, then 4 tied customers
  # (1 churner) competing for 1 remaining place in the top 2.
  actual <- c(1, 1, 0, 0, 0)
  score <- c(0.9, 0.5, 0.5, 0.5, 0.5)
  expect_equal(churners_in_top_k(actual, score, 2), 1 + 1 * (1 / 4))
})

test_that("gain table reaches 100% at full contact", {
  actual <- c(1, 0, 1, 0, 0, 1)
  expect_equal(tail(gain_table(actual, runif(6))$churners_captured_pct, 1), 100)
})

test_that("simulated data is reproducible and has a plausible churn rate", {
  a <- simulate_bank_customers(n = 2000, seed = 11)
  b <- simulate_bank_customers(n = 2000, seed = 11)
  expect_identical(a, b)
  expect_true(mean(a$churned) > 0.1 && mean(a$churned) < 0.35)
})

test_that("stratified split keeps the churn rate in both sets", {
  data <- simulate_bank_customers(n = 4000, seed = 5)
  s <- split_stratified(data, 0.7, seed = 5)
  expect_equal(nrow(s$train) + nrow(s$test), nrow(data))
  expect_lt(abs(mean(s$train$churned) - mean(s$test$churned)), 0.02)
})

test_that("logistic model recovers the planted direction of the main effects", {
  data <- simulate_bank_customers(n = 8000, seed = 99)
  fit <- fit_logistic(data)
  coefs <- coef(fit)
  expect_gt(coefs[["geographyGermany"]], 0)
  expect_gt(coefs[["is_activeinactive"]], 0)
  expect_gt(coefs[["complaintsyes"]], 0)
  expect_lt(coefs[["tenure_years"]], 0)
})

test_that("lasso predictions are probabilities of the right length", {
  data <- simulate_bank_customers(n = 1500, seed = 3)
  s <- split_stratified(data, 0.7, seed = 3)
  fit <- fit_lasso(s$train)
  p <- predict_lasso(fit, s$test)
  expect_length(p, nrow(s$test))
  expect_true(all(p > 0 & p < 1))
})
