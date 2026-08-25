# OpenLayout — apply the best macro-placement candidate's coordinates to the
# pristine floorplan checkpoint and write the macro_placed checkpoint.
# The flow renders `macro_coords` as a TCL list of {name x_um y_um orient}.

if {![info exists in_odb]}     { set in_odb ./saves_db/00_floorplan.odb }
if {![info exists out_odb]}    { set out_odb ./saves_db/01_macro_placed.odb }
if {![info exists data_dir]}   { set data_dir "." }
if {![info exists tech_dir]}   { set tech_dir ${data_dir} }
if {![info exists sdc_file]}   { set sdc_file ${data_dir}/constraint.sdc }
if {![info exists image_dir]}  { set image_dir ./images }
if {![info exists tag]}        { set tag macro_place }
if {![info exists macro_coords]} { set macro_coords {} }

# Fresh process: read_db restores tech/LEF but NOT liberty/SDC.
define_corners tt
read_liberty -corner tt ${tech_dir}/lib/NangateOpenCellLibrary_typical.lib
foreach _l [glob -nocomplain ${data_dir}/lib/*.lib] {
    if {[string match "*NangateOpenCellLibrary*" $_l]} continue
    read_liberty -corner tt $_l
}
read_db ${in_odb}
read_sdc ${sdc_file}

set _blk [ord::get_db_block]
set _dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set _n 0
foreach _e $macro_coords {
    lassign $_e _name _x _y _or
    set _inst [$_blk findInst $_name]
    if {$_inst eq "NULL" || $_inst eq ""} {
        # hierarchy separators may be stored escaped (\/) or plain (/)
        set _alt [string map {\\/ /} $_name]
        set _inst [$_blk findInst $_alt]
    }
    if {$_inst eq "NULL" || $_inst eq ""} {
        set _alt [string map {/ \\/} $_name]
        set _inst [$_blk findInst $_alt]
    }
    if {$_inst eq "NULL" || $_inst eq ""} {
        puts "OL_WARN macro $_name not found"
        continue
    }
    $_inst setOrient $_or
    $_inst setLocation [expr {int($_x * $_dbu)}] [expr {int($_y * $_dbu)}]
    $_inst setPlacementStatus FIRM
    incr _n
}
puts "OL_FACT macros_placed $_n"

write_db ${out_odb}
puts "OL_STAGE_DONE macro_place"
