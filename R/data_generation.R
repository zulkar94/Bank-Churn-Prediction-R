# Simulate a bank customer dataset with known churn drivers.
#
# The data is SYNTHETIC. The effects below are planted on purpose, so the
# models can be checked for whether they recover them. Real bank data would
# replace this function, with the same column names.

simulate_bank_customers <- function(n = 5000, seed = 2026) {
  set.seed(seed)

  age <- round(pmin(85, pmax(18, rnorm(n, mean = 42, sd = 12))))
  tenure_years <- sample(0:10, n, replace = TRUE)
  balance <- round(pmax(0, rnorm(n, mean = 60000, sd = 30000)) * rbinom(n, 1, 0.8))
  num_products <- sample(1:4, n, replace = TRUE, prob = c(0.50, 0.35, 0.10, 0.05))
  is_active <- rbinom(n, 1, 0.52)
  credit_score <- round(pmin(850, pmax(350, rnorm(n, mean = 650, sd = 90))))
  geography <- sample(c("France", "Germany", "Spain"), n, replace = TRUE,
                      prob = c(0.50, 0.25, 0.25))
  complaints <- rbinom(n, 1, 0.10)

  logit <- -2.6 +
    0.040 * (age - 42) +
    -0.070 * tenure_years +
    0.80 * (geography == "Germany") +
    0.85 * (1 - is_active) +
    -0.40 * (num_products == 2) +
    0.70 * (num_products >= 3) +
    1.20 * complaints +
    0.000003 * balance +
    -0.0015 * (credit_score - 650)

  churned <- rbinom(n, 1, stats::plogis(logit))

  data.frame(
    customer_id = sprintf("C%05d", seq_len(n)),
    age = age,
    tenure_years = tenure_years,
    balance = balance,
    num_products = factor(num_products),
    is_active = factor(is_active, levels = c(1, 0), labels = c("active", "inactive")),
    credit_score = credit_score,
    geography = factor(geography),
    complaints = factor(complaints, levels = c(0, 1), labels = c("no", "yes")),
    churned = churned,
    stringsAsFactors = FALSE
  )
}

# Stratified train/test split, so both sets keep the same churn rate.
split_stratified <- function(data, train_fraction = 0.7, seed = 2026) {
  set.seed(seed)
  train_index <- unlist(lapply(split(seq_len(nrow(data)), data$churned), function(idx) {
    sample(idx, size = round(length(idx) * train_fraction))
  }))
  list(
    train = data[sort(train_index), ],
    test = data[-sort(train_index), ]
  )
}
