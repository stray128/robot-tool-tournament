# Task

$task

# Training parts

$parts

# Placements

$placements

# Recipe parameters

$parameters

# Validator rules

A design that breaks any rule is rejected and never simulated.

$rules

# Engineering notes

$notes

# Distance rule

$distance_rule

# All $n_tested designs tested in this run, by cell

$tested

# Your team's best designs (top $n_elites of $n_pool, worst first, best last)

Score = the share of the $n_episodes training episodes ($n_parts parts x $n_placements placements x $n_seeds jitter seeds) that ended in a registered grasp; "lift" is the share that was lifted and held whatever the registration. Each design lists its score per part and per placement, how many episodes jammed on close or ended off centre, its failure modes, its mass and its largest piece count per jaw; "capture" is the generator's static prediction per part of the largest lateral error (mm) the jaws take without jamming and the largest they pull back to centre; "idea" is the hypothesis the design was made to test; "nearest" gives the closest tested designs in points.

$elites

# Informative failures

$failures

# Validator rejections in your last round

$rejections

# Target

$target

# Round $round of $last_round: what to submit

$operators
