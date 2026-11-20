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

# Lift in the top fraction of customers ranked by score.
# Lift of 2 means the top group churns at twice the overall rate.
lift_at <- function(actual, score, fraction = 0.2) {
  k <- ceiling(fraction * length(score))
  top <- order(score, decreasing = TRUE)[seq_len(k)]
  top_rate <- mean(actual[top])
  base_rate <- mean(actual)
  top_rate / base_rate
}
