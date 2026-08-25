# OpenLayout stage 3 — clock tree synthesis + post-CTS optimization
# (Innovus ccopt_design / optDesign -postCTS analogue: CTS, legalize,
# propagated clocks, repair_timing -setup / -hold, legalize again).
# Adapted from the proven cts_route Stage 03 script.

if {![info exists data_dir]}    { set data_dir "." }
if {![info exists tech_dir]}   { set tech_dir ${data_dir} }
if {![info exists proj_name]}   { set proj_name "design" }
if {![info exists sdc_file]}    { set sdc_file ${data_dir}/constraint.sdc }
if {![info exists save_dir]}    { set save_dir ./saves_db }
if {![info exists report_dir]}  { set report_dir ./reports }
if {![info exists image_dir]}   { set image_dir ./images }
if {![info exists tag]}         { set tag cts }
if {![info exists in_odb]}      { set in_odb ${save_dir}/02_detail_place.odb }
if {![info exists out_odb]}     { set out_odb ${save_dir}/03_cts.odb }
if {![info exists num_threads]} { set num_threads 8 }
# --- tunables ---
if {![info exists wire_unit]}        { set wire_unit 20 }
if {![info exists root_buf]}         { set root_buf CLKBUF_X3 }
if {![info exists buf_list]}         { set buf_list "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3" }
if {![info exists cluster_size]}     { set cluster_size 30 }
if {![info exists cluster_diameter]} { set cluster_diameter 100 }
if {![info exists repair_setup]}     { set repair_setup 1 }
if {![info exists repair_hold]}      { set repair_hold 1 }
if {![info exists hold_margin]}      { set hold_margin 0.05 }
file mkdir $save_dir $report_dir $image_dir

define_corners tt
read_liberty -corner tt ${tech_dir}/lib/NangateOpenCellLibrary_typical.lib
foreach _l [glob -nocomplain ${data_dir}/lib/*.lib] {
    if {[string match "*NangateOpenCellLibrary*" $_l]} continue
    read_liberty -corner tt $_l
}
read_db ${in_odb}
read_sdc ${sdc_file}
set_wire_rc -signal -layer metal3
set_wire_rc -clock  -layer metal5
set_thread_count $num_threads

# CTS must be free to buffer the clock nets (detail_place set dont_touch on
# them). Resolve clock-root nets from pins OR top-level ports.
proc clock_root_nets {} {
    set nets {}
    foreach clk [all_clocks] {
        set srcnames {}
        catch { foreach src [get_property $clk sources] { lappend srcnames [get_full_name $src] } }
        foreach pn $srcnames {
            set n [get_nets -quiet -of_objects [get_pins -quiet $pn]]
            if {$n eq ""} { set n [get_nets -quiet $pn] }
            foreach net $n { lappend nets [get_full_name $net] }
        }
    }
    return [lsort -unique $nets]
}
foreach _n [clock_root_nets] {
    catch { unset_dont_touch [get_nets $_n] }
    puts "unset_dont_touch clock net: $_n"
}

utl::report "Repair clock inverters"
repair_clock_inverters

utl::report "Clock Tree Synthesis"
clock_tree_synthesis -sink_clustering_enable \
                     -repair_clock_nets \
                     -buf_list $buf_list \
                     -root_buf $root_buf \
                     -sink_clustering_size $cluster_size \
                     -sink_clustering_max_diameter $cluster_diameter \
                     -wire_unit $wire_unit

utl::report "Detailed placement (legalize CTS buffers)"
detailed_placement
estimate_parasitics -placement

utl::report "Propagate clocks"
set_propagated_clock [all_clocks]

if {$repair_setup} {
    utl::report "Repair setup timing"
    repair_timing -setup -max_buffer_percent 20 -max_utilization 90
}
if {$repair_hold} {
    utl::report "Repair hold timing"
    repair_timing -hold -hold_margin $hold_margin
}

utl::report "Detailed placement (legalize repair cells)"
detailed_placement
utl::report "Check placement"
check_placement -verbose
estimate_parasitics -placement

utl::report "CTS report"
catch { report_cts -out_file ${report_dir}/${tag}.cts.rpt }
report_clock_skew    > ${report_dir}/${tag}.clock_skew.rpt
report_checks -path_delay max -group_count 10 > ${report_dir}/${tag}.setup.rpt
report_checks -path_delay min -group_count 10 > ${report_dir}/${tag}.hold.rpt
report_worst_slack -max
report_worst_slack -min
report_tns
report_design_area

write_db ${out_odb}
puts "OL_STAGE_DONE cts"
