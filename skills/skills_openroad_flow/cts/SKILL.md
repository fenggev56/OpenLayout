# OpenROAD CTS stage (+ post-CTS optimization)

Pipeline: `repair_clock_inverters` -> `clock_tree_synthesis` (TritonCTS,
buf_list/root_buf/sink clustering) -> `detailed_placement` ->
`set_propagated_clock [all_clocks]` -> `repair_timing -setup` ->
`repair_timing -hold -hold_margin M` (= optDesign -postCTS [-hold]) ->
`detailed_placement` -> `check_placement`.

Before CTS clocks are ideal; after `set_propagated_clock` real insertion
delays apply — hold is only meaningful from here on.
