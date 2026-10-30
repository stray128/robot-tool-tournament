"""The hosted runner: plan a robot job and run the Studio search on it, live, under hard caps.

A visitor on the static site describes a job. ``POST /plan`` asks the Task Engineer
(:mod:`rtt.studio.engineer`, the current prompt and rule set) for a plan. ``POST /runs`` sends a
validated plan to one Nebius Serverless Job, which runs the Studio search
(:func:`rtt.studio.search.run_task_tournament`) with the product's default settings and writes its
report and a compact summary to the results bucket. ``GET /runs/{id}`` reports progress and the
summary. The API is described in ``docs/RUNNER.md``.

Modules:

* :mod:`rtt.runner.config`: every cap and setting, read from the environment.
* :mod:`rtt.runner.store`: counters, the spend ledger and run records (DynamoDB, or in memory).
* :mod:`rtt.runner.caps`: the caps as store operations, the kill switch and the auto-off date.
* :mod:`rtt.runner.engineer`: the Task Engineer behind the three providers; keys stay in memory.
* :mod:`rtt.runner.nebius`: one Serverless Job per run, and the summary from Object Storage.
* :mod:`rtt.runner.summary`: the compact summary of a search report.
* :mod:`rtt.runner.app`: the HTTP router (Lambda Function URL events) and the run lifecycle.
* :mod:`rtt.runner.lambda_handler`: the Lambda entry point.
* :mod:`rtt.runner.local`: ``rtt runner serve`` on 127.0.0.1, against fakes or real keys.
* :mod:`rtt.runner.job`: the entry point inside the Nebius job (``python -m rtt.runner.job``).

Product mode: not part of the pre-registered experiment.
"""

from __future__ import annotations
