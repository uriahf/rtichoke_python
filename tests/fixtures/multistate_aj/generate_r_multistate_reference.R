#!/usr/bin/env Rscript
# Run from any directory; optional output filename supports regeneration tests.
for (pkg in c("survival", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Required R package missing: ", pkg)
}
states <- c("A", "B", "C")
panel <- data.frame(
  id = rep(1:4, each = 3), step = rep(0:2, 4),
  state = c("A", "B", "C", "A", "A", "B", "B", "B", "B", "B", "C", "C")
)
fit_case <- function(right_censored) {
  observed <- panel
  if (right_censored) observed <- subset(observed, !(id == 3 & step == 2))
  intervals <- do.call(rbind, lapply(split(observed, observed$id), function(subject) {
    n <- nrow(subject)
    data.frame(
      id = subject$id[-n], start = subject$step[-n], stop = subject$step[-1],
      initial = subject$state[-n],
      event = ifelse(subject$state[-n] == subject$state[-1], "censor", subject$state[-1])
    )
  }))
  # Censor subject 3 strictly between scheduled observations 1 and 2.
  if (right_censored) intervals <- rbind(intervals, data.frame(
    id = 3, start = 1, stop = 1.5, initial = "B", event = "censor"
  ))
  intervals$initial <- factor(intervals$initial, levels = states)
  intervals$event <- factor(intervals$event, levels = c("censor", states))
  rownames(intervals) <- NULL
  fit <- survival::survfit(
    survival::Surv(start, stop, event, type = "mstate") ~ 1,
    data = intervals, id = id, istate = initial, start.time = 0, time0 = TRUE
  )
  # Some survival versions store baseline separately in p0; normalize it into
  # the fitted pstate matrix using survival's own baseline helper.
  fit <- survival::survfit0(fit)
  stopifnot(identical(fit$states, states), min(fit$time) == 0)
  occupancy <- lapply(0:2, function(step) {
    index <- max(which(fit$time <= step))
    as.list(setNames(as.numeric(fit$pstate[index, ]), fit$states))
  })
  names(occupancy) <- as.character(0:2)
  list(panel = observed, intervals = intervals, occupancy = occupancy)
}
reference <- list(
  reference_package = "survival",
  version = as.character(utils::packageVersion("survival")),
  method = "survfit multistate Aalen-Johansen; pstate with empirical initial distribution",
  steps = 0:2, states = states,
  cases = list(complete = fit_case(FALSE), right_censored = fit_case(TRUE))
)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else file.path(dirname(script), "r_multistate_reference.json")
jsonlite::write_json(reference, output, pretty = TRUE, auto_unbox = TRUE, digits = 16)
cat("Generated", output, "using survival", reference$version, "\n")
