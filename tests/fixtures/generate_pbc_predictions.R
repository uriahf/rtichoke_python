# Recreate the held-out predictions from the R PBC overlay example.
# Pass the output CSV path as the first argument.
data("pbc", package = "survival")
cohort <- subset(pbc, complete.cases(age, bili, albumin, time, status))
cohort$time <- cohort$time / 365.25
set.seed(274)
training <- sample.int(nrow(cohort), floor(0.6 * nrow(cohort)))
train <- cohort[training, ]
test <- cohort[-training, ]
fit_rate <- function(cause, predictors) {
  train$event <- as.integer(train$status == cause)
  fit <- glm(reformulate(predictors, "event"), family = poisson(),
             offset = log(time), data = train)
  predict(fit, newdata = transform(test, time = 1), type = "response")
}
reals <- c(0L, 2L, 1L)[test$status + 1L]
models <- list(age_only = "age", age_and_labs = c("age", "log(bili)", "albumin"))
names(models) <- c("Age only", "Age + labs")
horizons <- c(1, 3, 5)
predictions <- lapply(models, function(predictors) {
  death <- fit_rate(2, predictors)
  transplant <- fit_rate(1, predictors)
  probs <- vapply(horizons, function(h) {
    death / (death + transplant) * -expm1(-(death + transplant) * h)
  }, numeric(nrow(test)))
  colnames(probs) <- as.character(horizons)
  probs
})
output <- data.frame(time = test$time, event = reals)
for (i in seq_along(models)) {
  for (h in horizons) {
    output[[paste(i, h, sep = "_")]] <- predictions[[i]][, as.character(h)]
  }
}
write.csv(output, commandArgs(trailingOnly = TRUE)[[1]], row.names = FALSE)
