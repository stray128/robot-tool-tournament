# Too similar to designs already tested

These new designs cannot be simulated. Each is a duplicate of, or closer than $min_points points to, a design that is already tested or already accepted this round$explore_clause:

$problems

Distances are in points: each number's range runs from 0 points (its minimum) to 100 points (its maximum), and a different pad or a different tip adds 100 points. Inactive numbers count at their fixed values. Designs in different cells (pad + tip) are therefore always at least 100 points apart. Cells that no tested or accepted design uses yet: $untried.

Call submit_designs again with every slot filled. Copy these slots unchanged: $keep. Replace only the listed slots, keeping each one's operator and idea where you can: either move the design to a cell that no design uses yet, or move its active numbers further from every design listed for it, by at least the shortfall plus 20 points.$fields_note
