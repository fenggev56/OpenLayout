# OpenROAD Routing stage (+ post-route optimization)

Pipeline: `set_global_routing_layer_adjustment` + `set_routing_layers` ->
`global_route` -> `estimate_parasitics -global_routing` -> `repair_design` /
`repair_timing -setup` / `repair_timing -hold` (= optDesign -postRoute
[-hold]; done BEFORE detailed route because the resizer invalidates routing)
-> `detailed_placement` -> re-`global_route` -> `detailed_route` -> final STA.
