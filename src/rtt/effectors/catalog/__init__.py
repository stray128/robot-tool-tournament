"""Shared helpers of the catalog families (product mode, not pre-registered).

* :mod:`rtt.effectors.catalog.vendored`: ``SOURCE.json`` licence records, verification, and
  loading a vendored MJCF as an attachable spec.
* :mod:`rtt.effectors.catalog.models`: the catalog table (mounting and synergies per model).
* :mod:`rtt.effectors.catalog.attach`: mounting a model palm-down on the shared wrist and the
  open-pose probe that measures its reach.
* :mod:`rtt.effectors.catalog.adaptive`: the procedural two- and three-finger adaptive
  grippers (our own geometry, tendon-coupled phalanges).

The families themselves are :mod:`rtt.effectors.catalog_hand` and
:mod:`rtt.effectors.catalog_gripper`.
"""
