---
name: openroad-optimizer
description: >
  Generic Optimizer strategy for the OpenROAD flow: how to learn from the
  attempt history, how to diversify parallel suggestions, and when to stop.
---

# OpenROAD Stage Optimization Strategy

## Method
1. **Read the history first.** Each attempt is params → metrics. Identify
   which parameter direction improved the failing metric and which hurt.
2. **Diversify.** Your K suggestions run in parallel and only the best is
   kept — never propose two near-identical sets. Good pattern: one
   conservative continuation of what worked + one different mechanism.
3. **One mechanism per knob change.** Congestion → lower density / raise
   layer_adjustment capacity / widen layer range. Setup timing → enable
   repair passes, timing_driven placement. Hold → hold_margin up. DRC →
   more droute iterations, less congestion. IR drop → smaller pdn_pitch.
4. **Never repeat** a parameter set that already ran (it is cached history,
   not new information).
5. **Respect ranges.** Out-of-range values are dropped by the system.

## When to stop
- The goal-relevant metrics for this stage are already met.
- The failing metric did not move over two rounds and no untried mechanism
  remains.
- The failure is caused by the input (e.g. impossible floorplan), not by
  this stage's parameters — say so in `reason`.
