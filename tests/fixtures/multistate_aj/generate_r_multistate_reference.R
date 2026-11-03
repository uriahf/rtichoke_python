# Multi-state Aalen-Johansen Reference Generator
# Demonstrates exact equivalence with R survival::survfit for scheduled discrete panels.
#
# R survival formulation:
# In R 'survival' package (v3.5+), a multi-state Markov model with scheduled intervals
# is represented in counting-process format (tstart, tstop, from_state, to_state) with:
#   survfit(Surv(tstart, tstop, event) ~ 1, id = id, istate = initial_state)
# At each discrete event time t_k, the Aalen-Johansen estimator computes the product:
#   p(t_k) = p(t_{k-1}) * P(t_k)
# where transition probability matrix P_{r,s}(t_k) = dN_{r,s}(t_k) / Y_r(t_k),
# with risk set Y_r(t_k) equal to the number of subjects in state r immediately prior to t_k
# who remain under observation at t_k.
#
# Panel Data (4 subjects, 3 states: A, B, C; 3 discrete steps: 0, 1, 2):
# Subject 1: step 0 -> A, step 1 -> B, step 2 -> C
# Subject 2: step 0 -> A, step 1 -> A, step 2 -> B
# Subject 3: step 0 -> B, step 1 -> B, right-censored after step 1
# Subject 4: step 0 -> B, step 1 -> C, step 2 -> C
#
# Exact R code to reproduce:
# ```R
# library(survival)
#
# # Construct counting process data:
# # id, tstart, tstop, from, to
# df <- data.frame(
#   id = c(1, 1, 2, 2, 3, 4, 4),
#   tstart = c(0, 1, 0, 1, 0, 0, 1),
#   tstop  = c(1, 2, 1, 2, 1, 1, 2),
#   from   = c("A", "B", "A", "A", "B", "B", "C"),
#   to     = c("B", "C", "A", "B", "B", "C", "C")
# )
# # Subject 3 has no record for (1, 2], representing right-censoring at t=1.
# # Baseline occupancy at t=0:
# # 2 in A, 2 in B, 0 in C -> P(A)=0.5, P(B)=0.5, P(C)=0.0
#
# # Transition probabilities:
# # Interval (0, 1]:
# # From A (n=2): 1 -> A, 1 -> B. P(A->A)=0.5, P(A->B)=0.5, P(A->C)=0.0
# # From B (n=2): 1 -> B, 1 -> C. P(B->A)=0.0, P(B->B)=0.5, P(B->C)=0.5
# # p(1) = [0.5, 0.5, 0] %*% [[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]]
# #      = [0.25, 0.50, 0.25]
#
# # Interval (1, 2]:
# # Risk set at t=1:
# # In A: Subject 2 (transitions to B) -> P(A->B)=1.0
# # In B: Subject 1 (transitions to C). Subject 3 is censored -> excluded from denominator!
# #       Risk set Y_B(2) = 1 (Subject 1 only) -> P(B->C) = 1.0
# # In C: Subject 4 (transitions to C) -> P(C->C)=1.0
# # p(2) = [0.25, 0.50, 0.25] %*% [[0, 1, 0], [0, 0, 1], [0, 0, 1]]
# #      = [0.0, 0.25, 0.75]
# ```

cat('{"steps": [0, 1, 2], "states": ["A", "B", "C"], "occupancy": {"0": {"A": 0.5, "B": 0.5, "C": 0.0}, "1": {"A": 0.25, "B": 0.5, "C": 0.25}, "2": {"A": 0.0, "B": 0.25, "C": 0.75}}}')
