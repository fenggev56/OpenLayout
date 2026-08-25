"""macroflow - structured multi-agent macro-placement flow.

Design split:

* LLM agents (Planner / Placer / Checker) make design decisions: macro
  strategy, placement shape, gate verdicts and revisions.
* Python modules handle system operations: OpenROAD process control,
  deterministic geometry, legalization, metric parsing, OpenROAD image capture,
  candidate bookkeeping, telemetry and TCL rendering.

The current orchestrator uses init, macro placement and fast-placement feedback.
It saves macro candidates and does not run the old detail+route optimization
loop.

This package is self-contained: it does not depend on the legacy ``openlayout``
agent framework. Sub-modules import lazily so the pure system-operation layer
can be tested without loading torch or starting OpenROAD.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
