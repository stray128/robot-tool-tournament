"""The v2 held-out runners (``freeform-v2``): H1, H2 and the secondary-family analysis.

Written after the ``prereg-v1`` tag, by the founder's decision of 2026-10-17
(``docs/DEVIATIONS.md``, "Open at the tag: the v2 held-out runners"), to the six rules of
``docs/PREREG.md`` section A6:

1. The seed is read only through :class:`~rtt.store.vault.HeldoutVault`, checked against
   the commitment in ``configs/suites/heldout_v1.sha256``
   (:func:`rtt.heldout_v2.procedural.run_heldout_v2`,
   :func:`rtt.heldout_v2.real.select_real_v2`). In a Nebius job it arrives as an injected
   file at ``/secret/heldout_seed`` (:data:`~rtt.heldout_v2.common.SEED_MOUNT`), in the
   held-out job only (:mod:`rtt.cloud.runs`).
2. Each run's frozen top-1 recipe (``frozen/<arm>-rNN.json``, hash-verified on load) is
   recompiled against every held-out part with the frozen generator
   (``rtt-freeform-gen/2``), built exactly as the search built it
   (:func:`~rtt.heldout_v2.common.build_one`).
3. ``success_v2`` is scored under ``task_v2`` by the same function as on ``train_v2``
   (:func:`rtt.sim.evaluate.evaluate_designs_v2`, reduced by
   :meth:`rtt.search.base.DesignScore.from_episodes_v2`): 1 ms step, native collision,
   ``physics_invalid`` and ``physics_unreliable`` as failures, the frozen registration
   tolerances, the suite's jitter seeds.
4. A recipe that builds invalid on a part scores ``invalid_build`` on every episode there.
5. H1 runs on ``heldout_procedural_v2`` drawn from the secret (100 conditions x 2 seeds);
   H2 on the scored scan draw (``heldout_real_v2``, 64 objects x 2 seeds), each object in
   the band its width across the jaw puts it in against the fixed 50 mm edge.
6. Every episode is written (``episodes.parquet``), with the run-level means the tests
   take and the lift-only means (exploratory).

Modules:

* :mod:`rtt.heldout_v2.common`: frozen picks, entrants, parallel builds, scoring, the code
  listing recorded in every report.
* :mod:`rtt.heldout_v2.procedural`: H1 (``rtt heldout --space freeform-v2``).
* :mod:`rtt.heldout_v2.real`: H2 (``rtt heldout-real {select,run} --space freeform-v2``).
* :mod:`rtt.heldout_v2.secondary`: S2, H1, H2 and Holm over the secondary family
  (PREREG A10; ``rtt analyze-secondary``).

Nothing in :mod:`rtt.search`, :mod:`rtt.sim`, :mod:`rtt.parts`, :mod:`rtt.design` or
:mod:`rtt.analysis` imports this package, and none of those frozen directories changed when
it was added (their code listings in PREREG A5 still hold).
"""
