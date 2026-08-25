# Optimizer knob map — route

| Problem | Knob |
|---|---|
| GRT overflow > 0 | layer_adjustment down (0.2), sig_max_layer up (metal9/10), congestion_iterations up |
| DRC count small, converging | droute_end_iter up (12-20) |
| DRC count large / shorts everywhere | fix congestion first, or go back to placement density |
| setup WNS < 0 | post_grt_repair_setup=1, post_grt_repair_design=1 |
| hold WNS < 0 | post_grt_repair_hold=1, hold_margin up |
| runtime explosion | droute_end_iter down, congestion_iterations down |

Very low layer_adjustment (<0.15) packs lower layers and can make detailed
route diverge — pair it with a higher sig_max_layer and expect longer runtime.
