# Optimizer strategy — finish (signoff ECO)

Choosing an upstream parameter re-runs the flow from the owning stage with
the previously successful scripts, then re-checks — each re-run costs a full
pass, so only do it when local analysis knobs cannot explain the problem.

| Signoff problem | Parameter (owning stage) |
|---|---|
| IR drop too high | pdn_pitch down (detail_place) |
| residual DRC | droute_end_iter up, or layer_adjustment down (route) |
| hold violations | hold_margin up (route) |
| setup WNS slightly negative | layer_adjustment down + droute_end_iter up (route) |
