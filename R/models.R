# Model fitting for churn: logistic regression, a class-weighted variant and
# a lasso-penalised logistic regression chosen by cross-validation.

churn_formula <- churned ~ age + tenure_years + balance + num_products +
  is_active + credit_score + geography + complaints

# Plain logistic regression.
fit_logistic <- function(train) {
  stats::glm(churn_formula, data = train, family = stats::binomial())
}

# Logistic regression with inverse-frequency weights, so the minority class
# (churners) has the same total weight as the majority class.
fit_logistic_weighted <- function(train) {
  # Weights live in a data column: glm() evaluates the formula in its own
  # environment, where a bare `weights` would resolve to stats::weights.
  p <- mean(train$churned)
  train$class_weight <- ifelse(train$churned == 1, 0.5 / p, 0.5 / (1 - p))
  # quasibinomial gives the same point estimates as binomial with non-integer
  # weights, without the "non-integer #successes" warning.
  stats::glm(churn_formula, data = train, family = stats::quasibinomial(),
             weights = class_weight)
}

# Lasso logistic regression. lambda is chosen by 10-fold cross-validation on the
# training data only, so the test set is never used for tuning.
fit_lasso <- function(train, seed = 2026) {
  x <- stats::model.matrix(churn_formula, data = train)[, -1, drop = FALSE]
  y <- train$churned
  set.seed(seed)
  glmnet::cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 10, type.measure = "auc")
}

predict_lasso <- function(fit, newdata) {
  x <- stats::model.matrix(stats::delete.response(stats::terms(churn_formula)), data = newdata)
  x <- x[, -1, drop = FALSE]
  as.numeric(stats::predict(fit, newx = x, s = "lambda.min", type = "response"))
}
