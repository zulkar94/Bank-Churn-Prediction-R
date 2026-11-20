# Evaluation metrics for a binary churn score.

# Area under the ROC curve, computed from ranks (Mann-Whitney U).
# Equals the probability that a random churner scores above a random non-churner.
auc_rank <- function(actual, score) {
  positives <- sum(actual == 1)
  negatives <- sum(actual == 0)
  if (positives == 0 || negatives == 0) {
    stop("AUC needs at least one positive and one negative case")
  }
  ranks <- rank(score, ties.method = "average")
  (sum(ranks[actual == 1]) - positives * (positives + 1) / 2) / (positives * negatives)
}

# Precision and recall at a probability threshold.
precision_recall <- function(actual, score, threshold = 0.5) {
  predicted <- score >= threshold
  true_pos <- sum(predicted & actual == 1)
  precision <- if (sum(predicted) == 0) NA_real_ else true_pos / sum(predicted)
  recall <- true_pos / sum(actual == 1)
  c(precision = precision, recall = recall)
}

# Expected number of churners among the k highest-scoring customers.
# Customers tied at the cut-off score share the remaining places equally, so the
# result does not depend on the row order of the data. Breaking ties by row
# order would let a model with no information look good if churners happen to
# come first.
churners_in_top_k <- function(actual, score, k) {
  if (k <= 0) return(0)
  if (k >= length(score)) return(sum(actual))
  cutoff <- sort(score, decreasing = TRUE)[k]
  above <- score > cutoff
  tied <- score == cutoff
  places_left <- k - sum(above)
  sum(actual[above]) + places_left * mean(actual[tied])
}

# Lift in the top fraction of customers ranked by score.
# Lift of 2 means the top group churns at twice the overall rate.
lift_at <- function(actual, score, fraction = 0.2) {
  k <- ceiling(fraction * length(score))
  top_rate <- churners_in_top_k(actual, score, k) / k
  top_rate / mean(actual)
}

# Share of all churners captured by contacting the top x% of customers.
gain_table <- function(actual, score, percents = seq(5, 100, by = 5)) {
  data.frame(
    contacted_pct = percents,
    churners_captured_pct = sapply(percents, function(p) {
      k <- ceiling(p / 100 * length(score))
      round(100 * churners_in_top_k(actual, score, k) / sum(actual), 1)
    })
  )
}
