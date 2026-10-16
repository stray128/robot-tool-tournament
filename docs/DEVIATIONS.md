# Deviations from the pre-registration

Every change to the protocol after the `prereg-v1` tag, and every rerun, is recorded here
with its date, what changed, why, and which results it can affect. Entries are added
before the affected result is looked at whenever that is possible. Results are never
dropped; a run that cannot complete is reported as failed.

Only infrastructure failures are rerun, with the same seed and the failed job's ID logged.

## Declared at registration

- **Gate G3 near-miss.** `task_v1` was frozen although calibration gate G3 failed on two
  criteria by narrow margins (the concave entrant at 75 % against a 70 % limit; 4 of 64
  random designs above the best entrant against 5 required). See `docs/TASK.md` and
  `docs/PREREG.md` section 9.

## After the tag

None yet.

<!--
Template:

### YYYY-MM-DD: short title

- **What changed:**
- **Why:**
- **Affects:** (endpoints, runs, job IDs)
- **Known before or after seeing the affected result:**
-->
