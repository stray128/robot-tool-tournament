"""Finger design: schema, 14-box generator, validators, mesh export and design distance.

* :mod:`rtt.design.types` - shared geometry contract with the simulator.
* :mod:`rtt.design.schema` - ``FingerDesign`` (10 parameters) and ``GripperTask``.
* :mod:`rtt.design.generate` - ``generate_finger`` and ``mass_and_inertia``.
* :mod:`rtt.design.validate` - ``validate`` with LLM-readable error messages.
* :mod:`rtt.design.export` - STL/GLB export of the box union.
* :mod:`rtt.design.distance` - normalised design distance for diversity checks.
"""
