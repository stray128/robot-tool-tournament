"""Real controllers and success predicates of the seven Studio primitives (track S2, ``rtt-effectors/2``).

One module per primitive registers its controller and predicate on import
(:func:`rtt.studio.plan.register_primitive`); :func:`rtt.studio.plan.get_primitive` imports
``rtt.studio.primitives.<primitive>`` on first use, so spawned workers find them.

* :mod:`.grasp`, :mod:`.suction`, :mod:`.wrap`: the three picks, adapted from the v1 family
  primitives and criteria through :class:`~rtt.studio.plan.V1SceneView` (:mod:`._pick`).
* :mod:`.hook_pull`, :mod:`.twist`, :mod:`.insert`, :mod:`.push`: the new primitives.
* :mod:`._common`: shared helpers (tool calibration, wrist poses, paths, contact readouts, the
  per-step schedule a controller shares with its predicate, the equal search budgets).

The rules every controller follows, their budgets, and every approximation are documented in
``docs/STUDIO_PRIMITIVES.md``. Importing this package registers nothing.
"""
