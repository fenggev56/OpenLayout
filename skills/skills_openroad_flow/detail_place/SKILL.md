# OpenROAD Placement stage (PDN + global + detailed)

Pipeline: fix macros (FIRM) -> cut_rows + macro halos -> PDN (`pdngen`,
core grid + macro power grid) -> `remove_buffers` + `repair_design` ->
`global_placement` -> `estimate_parasitics -placement` -> `repair_design` /
`repair_timing -setup` (= Innovus optDesign -preCTS) ->
`global_placement -routability_driven` -> `detailed_placement` +
`optimize_mirroring` -> `check_placement`.
