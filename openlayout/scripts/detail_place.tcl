# OpenLayout stage 2 — detailed placement (PDN + global place + resize +
# detailed place). Adapted from skills_Nangate45/2.detail_place, parameterized
# for the Optimizer agent (place_density, halo_um, pdn_pitch, ...).

if {![info exists data_dir]}    { set data_dir "." }
if {![info exists tech_dir]}   { set tech_dir ${data_dir} }
if {![info exists proj_name]}   { set proj_name "design" }
if {![info exists sdc_file]}    { set sdc_file ${data_dir}/constraint.sdc }
if {![info exists save_dir]}    { set save_dir ./saves_db }
if {![info exists report_dir]}  { set report_dir ./reports }
if {![info exists image_dir]}   { set image_dir ./images }
if {![info exists tag]}         { set tag detail_place }
if {![info exists in_odb]}      { set in_odb ${save_dir}/01_macro_placed.odb }
if {![info exists out_odb]}     { set out_odb ${save_dir}/02_detail_place.odb }
if {![info exists num_threads]} { set num_threads 8 }
# --- tunables ---
if {![info exists place_density]}      { set place_density 0.0 }
if {![info exists halo_um]}            { set halo_um 5.0 }
if {![info exists pdn_pitch]}          { set pdn_pitch 56.0 }
if {![info exists routability_driven]} { set routability_driven 1 }
if {![info exists timing_driven]}      { set timing_driven 1 }
if {![info exists run_resizer]}        { set run_resizer 1 }
file mkdir $save_dir $report_dir $image_dir

# Fresh process: re-establish liberty/SDC after read_db.
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

# Lock macros (pdngen requires FIXED macros).
set _nfix 0
foreach _inst [[ord::get_db_block] getInsts] {
    if {[[$_inst getMaster] isBlock]} { $_inst setPlacementStatus "FIRM"; incr _nfix }
}
utl::report "Fixed $_nfix macros (FIRM) before PDN"

# Trim rows around macros; sliver channels break pdngen (PDN-0179).
if {$_nfix > 0} {
    catch { cut_rows -halo_width_x 3.0 -halo_width_y 3.0 -row_min_width 12.0 }
}

# Placement-blockage halo around every macro (honored by GPL and DPL).
set _hblk [ord::get_db_block]
set _h [expr {int($halo_um * [$_hblk getDbUnitsPerMicron])}]
set _nhalo 0
foreach _inst [$_hblk getInsts] {
    if {![[$_inst getMaster] isBlock]} { continue }
    set _bb [$_inst getBBox]
    odb::dbBlockage_create $_hblk \
        [expr {[$_bb xMin] - $_h}] [expr {[$_bb yMin] - $_h}] \
        [expr {[$_bb xMax] + $_h}] [expr {[$_bb yMax] + $_h}]
    incr _nhalo
}
utl::report "Added ${halo_um}um placement halo around $_nhalo macros"

##########################################################################
# Global connections + PDN (Nangate45 core-only stripes)
##########################################################################
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDD} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VSS} -ground
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDDARRAY} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDDARRAY!} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDD!} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VSS!} -ground
global_connect

set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
define_pdn_grid -name {core_grid} -voltage_domains {CORE}

set pgcrWidth    1.6
set pgcrSpacing  1.6
set pgcrOffset   2.0
set tpg1Width    1.6
set tpg1Pitch    $pdn_pitch
set tpg1Spacing  2.0
set tpg1Offset  20.0
set tpg4Width    0.96
set tpg4Pitch    $pdn_pitch
set tpg4Spacing  2.0
set tpg4Offset  20.0

# Core ring only if the core-to-die margin fits it (PDN-0351 otherwise).
set block [ord::get_db_block]
set _ring_need [expr {$pgcrOffset + 2*$pgcrWidth + $pgcrSpacing}]
set _dbu  [[ord::get_db_tech] getDbUnitsPerMicron]
set _die  [$block getDieArea]
set _core [$block getCoreArea]
set _mgn [expr {min( \
    ([$_core xMin]-[$_die xMin]), ([$_die xMax]-[$_core xMax]), \
    ([$_core yMin]-[$_die yMin]), ([$_die yMax]-[$_core yMax]) ) / double($_dbu)}]
set _ext "-extend_to_core_ring"
if {$_mgn < $_ring_need + 0.2} {
    set _ext ""
    utl::report "Margin ${_mgn}um < ring need ${_ring_need}um: stripes-only PDN"
} else {
    utl::report "Margin ${_mgn}um: adding core ring"
    add_pdn_ring -grid {core_grid} -layer {metal9 metal10} \
       -widths "$pgcrWidth $pgcrWidth" -spacings "$pgcrSpacing $pgcrSpacing" \
       -core_offsets "$pgcrOffset $pgcrOffset" -add_connect
}
add_pdn_stripe -grid {core_grid} -layer {metal1} -width 0.17 -offset 0 \
               -followpins {*}$_ext
add_pdn_stripe -grid {core_grid} -layer {metal8} -width $tpg4Width \
               -pitch $tpg4Pitch -spacing $tpg4Spacing -offset $tpg4Offset \
               {*}$_ext -snap_to_grid
add_pdn_stripe -grid {core_grid} -layer {metal9} -width $tpg1Width \
               -pitch $tpg1Pitch -spacing $tpg1Spacing -offset $tpg1Offset \
               {*}$_ext -snap_to_grid
add_pdn_connect -grid {core_grid} -layers {metal9 metal8}
add_pdn_connect -grid {core_grid} -layers {metal8 metal1}
# ----- Macro power grid: connect macro power pins (metal4 strips on the
# fakerams) up into the core grid so IR analysis (psm) sees a connected net.
if {$_nfix > 0} {
    define_pdn_grid -macro -default -name {macro_grid} \
        -voltage_domains {CORE} -halo "2.0 2.0"
    # Cross-hatch straps (m5 H + m6 V) at tight pitch so every macro power
    # pin strip is crossed regardless of macro orientation/size — a single
    # m5 layer at pitch 20 missed a rotated 64x7 fakeram's ~31um pins.
    add_pdn_stripe -grid {macro_grid} -layer {metal5} -width 0.93 \
        -pitch 10.0 -spacing 2.0 -offset 2.0 -snap_to_grid
    add_pdn_stripe -grid {macro_grid} -layer {metal6} -width 0.93 \
        -pitch 10.0 -spacing 2.0 -offset 2.0 -snap_to_grid
    add_pdn_connect -grid {macro_grid} -layers {metal4 metal5}
    add_pdn_connect -grid {macro_grid} -layers {metal4 metal6}
    add_pdn_connect -grid {macro_grid} -layers {metal5 metal6}
    add_pdn_connect -grid {macro_grid} -layers {metal6 metal8}
}

pdngen

##########################################################################
# Protect clock-root nets (CTS-0114 otherwise), then place
##########################################################################
if {[llength [all_clocks]] == 0} { catch { read_sdc ${sdc_file} } }
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
set CLK_NETS [clock_root_nets]
if {[llength $CLK_NETS]} {
    catch { set_dont_touch [get_nets $CLK_NETS] }
    puts "DONT_TOUCH clk nets: $CLK_NETS"
}

utl::report "Remove buffers"
remove_buffers
utl::report "Repair design (pre-place)"
repair_design

# density: 0.0 => auto (uniform density + margin)
set _gp_args {}
if {$place_density > 0.0} { lappend _gp_args -density $place_density }
if {$timing_driven}       { lappend _gp_args -timing_driven }

utl::report "Global Placement (1)"
global_placement {*}$_gp_args

utl::report "Estimate parasitics"
estimate_parasitics -placement
if {$run_resizer} {
    utl::report "Repair design"
    repair_design
    set REPAIR_BOUNDS "-max_buffer_percent 20 -max_utilization 90 -max_passes 50"
    utl::report "Repair setup"
    repair_timing -setup {*}$REPAIR_BOUNDS
}

utl::report "Global Placement (2)"
set _gp2_args $_gp_args
if {$routability_driven} {
    lappend _gp2_args -routability_driven -routability_check_overflow 0.30
}
global_placement {*}$_gp2_args

utl::report "Detailed placement"
detailed_placement
optimize_mirroring
estimate_parasitics -placement

utl::report "Check placement"
check_placement -verbose -report_file_name ${report_dir}/${tag}.place_check.rpt

report_worst_slack -max
report_worst_slack -min
report_tns
report_design_area

write_db ${out_odb}
puts "OL_STAGE_DONE detail_place"
