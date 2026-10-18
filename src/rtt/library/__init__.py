"""The open effector library (product mode, not part of the pre-registered experiment).

Open-source end-effectors whose files may be redistributed, each pinned to an upstream
commit, hashed, licence-gated and given a simulation status:

* :mod:`rtt.library.licence`: the licence gate (green, amber, red lanes; the chain rule;
  credit lines);
* :mod:`rtt.library.taxonomy`: capability tags, kinds, classes and sim statuses;
* :mod:`rtt.library.mounts`: ISO 9409-1 flange patterns (with the provenance of every
  dimension) and item mount records;
* :mod:`rtt.library.brands`: the brand guard for on-screen text;
* :mod:`rtt.library.records`: ``SOURCE.json`` per item, ``lock.json``, verification;
* :mod:`rtt.library.fetch`, :mod:`rtt.library.ingest`: pinned-commit vendoring
  (``scripts/vendor_library.py``) with licence-drift detection;
* :mod:`rtt.library.proxy`, :mod:`rtt.library.preview`: rigid simulation proxies (CoACD) and
  the catalog page's GLB previews;
* :mod:`rtt.library.adapters`: the parametric ISO 9409-1 adapter plate generator;
* :mod:`rtt.library.entrants`: items as effector entrants of the existing families, and the
  compile and attach gate;
* :mod:`rtt.library.web`: the catalog page's data.

``docs/LIBRARY.md`` describes the policy and the ingest; the founder's decisions of 2026-10-18
are recorded there (amber items excluded; two industrial grippers' simulation models included,
their makers named only in credits).
"""

from rtt.library.licence import (
    GREEN_LICENCES,
    ChainLink,
    Rating,
    attribution_line,
    chain_rating,
    classify,
    gate,
)

__all__ = ["GREEN_LICENCES", "ChainLink", "Rating", "attribution_line", "chain_rating", "classify", "gate"]
