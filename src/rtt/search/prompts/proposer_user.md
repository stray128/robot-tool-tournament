# Task

$task

# Design parameters

$parameters

# Validator rules

A design that breaks any rule is rejected and never simulated.

$rules

# Tested designs (top $n_elites of $n_pool, worst first, best last)

Score = success rate over the $n_conditions training conditions at the design's best controller setting k. The per-family success, failure-mode counts and finger mass are at that setting; "idea" is the hypothesis the design was made to test.

$elites

# Informative failures

$failures

# Validator rejections in your last round

$rejections

# Target

$target

# Round $round of $last_round: what to submit

Call submit_designs with exactly 4 new designs, one per slot:
- mutate_1 and mutate_2: each starts from ONE tested design (put its label, e.g. "d05", in parents) and makes one bold change aimed at that design's main failure mode.
- crossover: combines features of TWO tested designs (both labels in parents).
- explore: a clearly different design that tests an idea none of the tested designs covers.

$distance_rule
