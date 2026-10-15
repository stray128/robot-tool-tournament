"""End-to-end workflows built from the design, search, simulator and store packages.

* :mod:`rtt.pipeline.replays` - re-simulate scored episodes with recording on, as web replays.
* :mod:`rtt.pipeline.slice` - the vertical slice: Nemotron proposes, the simulator scores,
  replays and a manifest are written (``rtt slice``).
* :mod:`rtt.pipeline.heldout` - evaluate frozen designs on the commit-reveal held-out suite
  (``rtt heldout``).

Each module imports MuJoCo lazily through :mod:`rtt.sim`, so importing this package is cheap.
"""
