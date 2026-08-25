# OpenLayout stage 5 — finish/signoff: IR drop analysis (real solve via
# analyze_power_grid, adapted in English from PV_agent run_ir_drop_real.tcl)
# plus the final PPA / wirelength / timing reports on the routed database.
# IR drop can only be checked after routing; if it is bad, the Optimizer
# refines upstream parameters (e.g. pdn_pitch) and reuses the previous
# successful scripts.

if {![info exists data_dir]}    { set data_dir "." }
if {![info exists tech_dir]}   { set tech_dir ${data_dir} }
if {![info exists proj_name]}   { set proj_name "design" }
if {![info exists sdc_file]}    { set sdc_file ${data_dir}/constraint.sdc }
if {![info exists save_dir]}    { set save_dir ./saves_db }
if {![info exists report_dir]}  { set report_dir ./reports }
if {![info exists image_dir]}   { set image_dir ./images }
if {![info exists tag]}         { set tag finish }
if {![info exists in_odb]}      { set in_odb ${save_dir}/04_route.odb }
if {![info exists out_odb]}     { set out_odb ${save_dir}/05_finish.odb }
if {![info exists num_threads]} { set num_threads 8 }
# --- tunables ---
if {![info exists ir_source_type]} { set ir_source_type FULL }
if {![info exists vdd_voltage]}    { set vdd_voltage 1.1 }
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
set_propagated_clock [all_clocks]
estimate_parasitics -global_routing

puts "============================================"
puts "Final timing / area / wirelength / power"
puts "============================================"
report_worst_slack -max
report_worst_slack -min
report_tns
report_design_area
# (no report_wire_length: needs -net on this build; route stage logged WL)
report_power > ${report_dir}/${tag}.power.rpt
report_power
report_checks -path_delay max -group_count 10 > ${report_dir}/${tag}.setup.rpt
report_checks -path_delay min -group_count 10 > ${report_dir}/${tag}.hold.rpt

puts "============================================"
puts "IR drop analysis (analyze_power_grid)"
puts "============================================"
# Set supply voltages
set_pdnsim_net_voltage -net VDD -voltage $vdd_voltage
set_pdnsim_net_voltage -net VSS -voltage 0.0

# Analyze VDD power grid
puts "=== Analyzing VDD power grid ==="
if {[catch {
    analyze_power_grid -net VDD \
        -source_type $ir_source_type \
        -voltage_file ${report_dir}/${tag}.vdd_voltage.rpt \
        -error_file   ${report_dir}/${tag}.vdd_pg_error.rpt
} _err]} {
    puts "OL_WARN VDD IR analysis failed: $_err"
}

# Analyze VSS power grid
puts "=== Analyzing VSS power grid ==="
if {[catch {
    analyze_power_grid -net VSS \
        -source_type $ir_source_type \
        -voltage_file ${report_dir}/${tag}.vss_voltage.rpt \
        -error_file   ${report_dir}/${tag}.vss_pg_error.rpt
} _err]} {
    puts "OL_WARN VSS IR analysis failed: $_err"
}

write_db ${out_odb}
write_def ${save_dir}/${proj_name}_final.def
write_verilog ${save_dir}/${proj_name}_final.v
puts "OL_STAGE_DONE finish"
