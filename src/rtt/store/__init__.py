"""Result storage: per-run logs and Parquet tables, run manifests, and the held-out seed vault.

* :mod:`rtt.store.runs` - JSONL logs and Parquet tables of designs, episodes and rejections.
* :mod:`rtt.store.manifest` - job, code, model and cost facts for every search job.
* :mod:`rtt.store.vault` - commit-reveal held-out seed and the guard that keeps search away from it.
"""
