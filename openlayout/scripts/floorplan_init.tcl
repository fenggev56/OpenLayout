# OpenLayout stage 0 — init/floorplan.
# Loads libs/LEFs/netlist and the PREDEFINED floorplan DEF, sets layer RC,
# prints design facts (macro count, cells, core area, clocks) and writes the
# pristine floorplan checkpoint. Adapted from skills_Nangate45/0.init.

if {![info exists data_dir]}     { set data_dir "." }
if {![info exists tech_dir]}     { set tech_dir ${data_dir} }
if {![info exists proj_name]}    { set proj_name "design" }
if {![info exists top_design]}   { set top_design $proj_name }
if {![info exists netlist_file]} { set netlist_file ${data_dir}/${proj_name}.v }
if {![info exists sdc_file]}     { set sdc_file ${data_dir}/constraint.sdc }
if {![info exists fp_def]}       { set fp_def ${data_dir}/def/1_1_u60_floorplan.def }
if {![info exists save_dir]}     { set save_dir ./saves_db }
if {![info exists report_dir]}   { set report_dir ./reports }
if {![info exists image_dir]}    { set image_dir ./images }
if {![info exists tag]}          { set tag floorplan }
if {![info exists out_odb]}      { set out_odb ${save_dir}/00_floorplan.odb }
file mkdir $save_dir $report_dir $image_dir

# PDK (std cells + tech LEF) from tech_dir; design macros from data_dir.
set pdk_cells_lib ${tech_dir}/lib
set pdk_cells_lef ${tech_dir}/lef
set design_lib    ${data_dir}/lib
set design_lef    ${data_dir}/lef

define_corners tt
puts "Init standard cells"
read_liberty -corner tt ${pdk_cells_lib}/NangateOpenCellLibrary_typical.lib

puts "Init tech-lef"
read_lef ${pdk_cells_lef}/NangateOpenCellLibrary.tech.lef
puts "Init cell-lef"
read_lef ${pdk_cells_lef}/NangateOpenCellLibrary.macro.mod.lef

# Every non-stdcell LEF/LIB in the circuit's dirs (fakeram + custom macros).
puts "Init macro lef"
foreach _lef [lsort [glob -nocomplain ${design_lef}/*.lef]] {
    if {[string match "*NangateOpenCellLibrary*" $_lef]} { continue }
    puts "  read_lef $_lef"
    read_lef $_lef
}
puts "Init macro liberty"
foreach _lib [lsort [glob -nocomplain ${design_lib}/*.lib]] {
    if {[string match "*NangateOpenCellLibrary*" $_lib]} { continue }
    puts "  read_liberty $_lib"
    read_liberty -corner tt $_lib
}

read_verilog ${netlist_file}
read_def ${fp_def}
read_sdc ${sdc_file}

# Nangate45 RC (liberty units fF,kOhm). Must come AFTER the netlist is linked.
set_layer_rc -layer metal1 -resistance 5.4286e-03 -capacitance 7.41819E-02
set_layer_rc -layer metal2 -resistance 3.5714e-03 -capacitance 6.74606E-02
set_layer_rc -layer metal3 -resistance 3.5714e-03 -capacitance 8.88758E-02
set_layer_rc -layer metal4 -resistance 1.5000e-03 -capacitance 1.07121E-01
set_layer_rc -layer metal5 -resistance 1.5000e-03 -capacitance 1.08964E-01
set_layer_rc -layer metal6 -resistance 1.5000e-03 -capacitance 1.02044E-01
set_layer_rc -layer metal7 -resistance 1.8750e-04 -capacitance 1.10436E-01
set_layer_rc -layer metal8 -resistance 1.8750e-04 -capacitance 9.69714E-02
set_wire_rc -signal -layer metal3
set_wire_rc -clock  -layer metal5

# ---- design facts (parsed by the flow into the design context) ----
set _blk [ord::get_db_block]
set _nmacro 0
set _ninst 0
foreach _i [$_blk getInsts] {
    incr _ninst
    if {[[$_i getMaster] isBlock]} { incr _nmacro }
}
set _dbu  [[ord::get_db_tech] getDbUnitsPerMicron]
set _core [$_blk getCoreArea]
set _cw [expr {([$_core xMax]-[$_core xMin])/double($_dbu)}]
set _ch [expr {([$_core yMax]-[$_core yMin])/double($_dbu)}]
puts "OL_FACT num_instances $_ninst"
puts "OL_FACT num_macros $_nmacro"
puts "OL_FACT core_width_um [format %.2f $_cw]"
puts "OL_FACT core_height_um [format %.2f $_ch]"
puts "OL_FACT core_bbox_um [format %.3f [expr {[$_core xMin]/double($_dbu)}]] [format %.3f [expr {[$_core yMin]/double($_dbu)}]] [format %.3f [expr {[$_core xMax]/double($_dbu)}]] [format %.3f [expr {[$_core yMax]/double($_dbu)}]]"
# macro sizes (unoriented master W H) for the Python macro legalizer
foreach _i [$_blk getInsts] {
    set _m [$_i getMaster]
    if {[$_m isBlock]} {
        puts "OL_MACRO [$_i getName] [format %.3f [expr {[$_m getWidth]/double($_dbu)}]] [format %.3f [expr {[$_m getHeight]/double($_dbu)}]]"
    }
}
foreach _clk [all_clocks] {
    puts "OL_FACT clock [get_name $_clk] period_ns [get_property $_clk period]"
}
report_design_area

write_db ${out_odb}
puts "OL_STAGE_DONE floorplan"
