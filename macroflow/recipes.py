"""TCL recipes — small, read-mostly snippets Python sends to OpenROAD.

These are the *queries and primitives* the system-operation layer relies on
(die/core size, macro list, pin boxes, the ``safe_place`` proc, overlap check,
stdcell-density metric, the congestion proxy, and the fresh-process route
preamble). They are intentionally separate from the big per-stage *templates*
(floorplan / detail_place / cts_route), which live under ``skills/`` and are
loaded through :mod:`macroflow.skills`.

Recipes are proven verbatim from the validated classic flow.
"""

from __future__ import annotations

# Die + core dimensions in microns (the placement canvas).
DIE_CORE = r"""
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
set die   [$block getDieArea]
set core  [$block getCoreArea]

set die_w  [expr ([$die  xMax] - [$die  xMin]) / double($dbu)]
set die_h  [expr ([$die  yMax] - [$die  yMin]) / double($dbu)]
set die_x  [expr [$die  xMin] / double($dbu)]
set die_y  [expr [$die  yMin] / double($dbu)]
set core_w [expr ([$core xMax] - [$core xMin]) / double($dbu)]
set core_h [expr ([$core yMax] - [$core yMin]) / double($dbu)]
set core_x [expr [$core xMin] / double($dbu)]
set core_y [expr [$core yMin] / double($dbu)]

puts [format "Die  width  = %.3f um" $die_w]
puts [format "Die  height = %.3f um" $die_h]
puts [format "Die  origin = (%.3f, %.3f) um" $die_x $die_y]
puts [format "Die  x_max  = %.3f um" [expr $die_x + $die_w]]
puts [format "Die  y_max  = %.3f um" [expr $die_y + $die_h]]
puts [format "Core width  = %.3f um" $core_w]
puts [format "Core height = %.3f um" $core_h]
puts [format "Core origin = (%.3f, %.3f) um" $core_x $core_y]
puts [format "Core x_max  = %.3f um" [expr $core_x + $core_w]]
puts [format "Core y_max  = %.3f um" [expr $core_y + $core_h]]
"""

# Every macro instance (master isBlock) with size + current origin.
MACRO_LIST = r"""
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]

foreach inst [$block getInsts] {
    set master [$inst getMaster]
    if {[$master isBlock]} {
        set w [expr [$master getWidth]  / double($dbu)]
        set h [expr [$master getHeight] / double($dbu)]
        set origin [$inst getOrigin]
        set x [expr [lindex $origin 0] / double($dbu)]
        set y [expr [lindex $origin 1] / double($dbu)]
        set orient [$inst getOrient]
        puts [format "%-55s  master=%-20s  size=%.3f x %.3f  pos=(%.3f, %.3f)  orient=%s" \
            [$inst getName] [$master getName] $w $h $x $y $orient]
    }
}
"""

# IO pin (block terminal) boxes for image bounds/overlay. These are not passed
# to the model as text.
PIN_COORDS = r"""
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
foreach bterm [$block getBTerms] {
    foreach bpin [$bterm getBPins] {
        foreach box [$bpin getBoxes] {
            set x0 [expr [$box xMin] / double($dbu)]
            set y0 [expr [$box yMin] / double($dbu)]
            set x1 [expr [$box xMax] / double($dbu)]
            set y1 [expr [$box yMax] / double($dbu)]
            puts [format "PIN_BOX box=(%.3f, %.3f, %.3f, %.3f)" $x0 $y0 $x1 $y1]
        }
    }
}
"""

# Mark each macro's dominant signal-pin side with GUI rulers and print a compact
# summary. getITerm/getBBox returns transformed pin boxes for the currently
# placed orientation, so the side here is the side visible in the saved image.
MACRO_PIN_SIDES = r"""
catch {gui::clear_rulers}
set block [ord::get_db_block]
set dbu [[ord::get_db_tech] getDbUnitsPerMicron]

foreach inst [$block getInsts] {
    set master [$inst getMaster]
    if {![$master isBlock]} { continue }

    set bbox [$inst getBBox]
    set xmin [$bbox xMin]
    set ymin [$bbox yMin]
    set xmax [$bbox xMax]
    set ymax [$bbox yMax]

    set left 0
    set right 0
    set top 0
    set bottom 0
    foreach iterm [$inst getITerms] {
        if {[catch {$iterm getSigType} sigtype] || $sigtype ne "SIGNAL"} { continue }
        if {[catch {$iterm getBBox} pbb] || $pbb eq "NULL" || $pbb eq ""} { continue }
        set px [expr {([$pbb xMin] + [$pbb xMax]) / 2.0}]
        set py [expr {([$pbb yMin] + [$pbb yMax]) / 2.0}]
        set dl [expr {$px - $xmin}]
        set dr [expr {$xmax - $px}]
        set db [expr {$py - $ymin}]
        set dt [expr {$ymax - $py}]
        set mind [expr {min($dl, $dr, $db, $dt)}]
        if {$mind == $dl} {
            incr left
        } elseif {$mind == $dr} {
            incr right
        } elseif {$mind == $db} {
            incr bottom
        } else {
            incr top
        }
    }

    set maxc $left
    set side "L"
    if {$right > $maxc} { set maxc $right; set side "R" }
    if {$top > $maxc} { set maxc $top; set side "T" }
    if {$bottom > $maxc} { set maxc $bottom; set side "B" }
    if {$maxc <= 0} { set side "NONE" }

    set xmin_um [expr {$xmin / double($dbu)}]
    set ymin_um [expr {$ymin / double($dbu)}]
    set xmax_um [expr {$xmax / double($dbu)}]
    set ymax_um [expr {$ymax / double($dbu)}]
    set iname [$inst getName]
    set orient [$inst getOrient]

    if {$side eq "L"} {
        catch {gui::add_ruler $xmin_um $ymin_um $xmin_um $ymax_um "PIN" "${iname}_pin" false}
    } elseif {$side eq "R"} {
        catch {gui::add_ruler $xmax_um $ymin_um $xmax_um $ymax_um "PIN" "${iname}_pin" false}
    } elseif {$side eq "B"} {
        catch {gui::add_ruler $xmin_um $ymin_um $xmax_um $ymin_um "PIN" "${iname}_pin" false}
    } elseif {$side eq "T"} {
        catch {gui::add_ruler $xmin_um $ymax_um $xmax_um $ymax_um "PIN" "${iname}_pin" false}
    }

    puts [format "MACRO_PIN_SIDE name=%s orient=%s side=%s dominant=%d counts=L:%d,R:%d,T:%d,B:%d bbox=(%.3f,%.3f,%.3f,%.3f)" \
        $iname $orient $side $maxc $left $right $top $bottom \
        $xmin_um $ymin_um $xmax_um $ymax_um]
}
"""

# Place-one-macro proc. Unlock to NONE first (place_macro is a silent no-op on
# fixed insts). Names are passed brace-quoted by the caller so escaped bus
# brackets (GEN_MACRO_11x2\[8\]) reach findInst verbatim.
SAFE_PLACE_PROC = r"""
proc safe_place {name x y {orient R0}} {
    set inst [[ord::get_db_block] findInst $name]
    if {$inst eq "NULL" || $inst eq ""} {
        puts "SKIP: macro '$name' not found"
        return
    }
    $inst setPlacementStatus "NONE"
    if {[catch {place_macro -macro_name $name -location [list $x $y] -orient $orient} err]} {
        puts "WARN: $name placement failed: $err"
    } else {
        puts "OK: $name placed at ($x, $y) orient $orient"
    }
}
puts "safe_place proc loaded"
"""

LOCK_MACROS_FIRM = r"""
set _nfirm 0
foreach _inst [[ord::get_db_block] getInsts] {
    if {[[$_inst getMaster] isBlock]} {
        $_inst setPlacementStatus "FIRM"
        incr _nfirm
    }
}
puts "LOCK_MACROS_FIRM: $_nfirm"
"""

# Count overlapping macro pairs (legalizer sanity check before detail place).
CHECK_OVERLAPS = r"""
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
set macros {}
foreach inst [$block getInsts] {
    set master [$inst getMaster]
    if {[$master isBlock]} {
        set bb [$inst getBBox]
        set x  [$bb xMin]
        set y  [$bb yMin]
        set x2 [$bb xMax]
        set y2 [$bb yMax]
        lappend macros [list [$inst getName] $x $y $x2 $y2]
    }
}
set n [llength $macros]
set overlaps 0
for {set i 0} {$i < $n} {incr i} {
    for {set j [expr {$i+1}]} {$j < $n} {incr j} {
        set a [lindex $macros $i]
        set b [lindex $macros $j]
        if {[lindex $a 1] < [lindex $b 3] && [lindex $a 3] > [lindex $b 1] && \
            [lindex $a 2] < [lindex $b 4] && [lindex $a 4] > [lindex $b 2]} {
            puts [format "OVERLAP: %s  <->  %s" [lindex $a 0] [lindex $b 0]]
            incr overlaps
        }
    }
}
puts "Total overlaps: $overlaps"
"""

CHECK_MACROS_IN_CORE = r"""
set block [ord::get_db_block]
set core [$block getCoreArea]
set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set tol [expr {int(0.10 * $dbu)}]
set outside 0
foreach inst [$block getInsts] {
    set master [$inst getMaster]
    if {[$master isBlock]} {
        set bb [$inst getBBox]
        if {[$bb xMin] < [expr {[$core xMin] - $tol}] || \
            [$bb yMin] < [expr {[$core yMin] - $tol}] || \
            [$bb xMax] > [expr {[$core xMax] + $tol}] || \
            [$bb yMax] > [expr {[$core yMax] + $tol}]} {
            puts [format "OUTSIDE_CORE: %s bbox=(%d,%d,%d,%d)" \
                [$inst getName] [$bb xMin] [$bb yMin] [$bb xMax] [$bb yMax]]
            incr outside
        }
    }
}
puts "Total outside core: $outside"
"""

# Quantitative stdcell-spread metric for the Gate-B convergence test: fraction
# of the stdcell bounding box actually occupied by cells. Higher => tighter.
STDCELL_DENSITY = r"""
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
set minx 1e18; set miny 1e18; set maxx -1e18; set maxy -1e18
set cell_area 0.0
set n 0
foreach inst [$block getInsts] {
    set master [$inst getMaster]
    if {[$master isBlock]} { continue }
    set st [$inst getPlacementStatus]
    if {$st eq "NONE" || $st eq "UNPLACED"} { continue }
    set bb [$inst getBBox]
    set x0 [$bb xMin]; set y0 [$bb yMin]
    set x1 [$bb xMax]; set y1 [$bb yMax]
    if {$x0 < $minx} { set minx $x0 }
    if {$y0 < $miny} { set miny $y0 }
    if {$x1 > $maxx} { set maxx $x1 }
    if {$y1 > $maxy} { set maxy $y1 }
    set cell_area [expr {$cell_area + [$master getWidth] * [$master getHeight]}]
    incr n
}
if {$n == 0 || $maxx <= $minx || $maxy <= $miny} {
    puts "STDCELL_DENSITY: 0.0"
    puts "STDCELL_BBOX_AREA: 0.0"
} else {
    set d2 [expr {double($dbu) * double($dbu)}]
    set bbox_area [expr {($maxx - $minx) * ($maxy - $miny) / $d2}]
    set ca [expr {$cell_area / $d2}]
    set density [expr {$ca / $bbox_area}]
    puts [format "STDCELL_CELLS: %d" $n]
    puts [format "STDCELL_AREA: %.3f" $ca]
    puts [format "STDCELL_BBOX_AREA: %.3f" $bbox_area]
    puts [format "STDCELL_DENSITY: %.4f" $density]
}
"""

# Fast congestion + wirelength estimate via global routing only. Settings match
# the light final route (capacity 0.3, signal metal2-metal9, 32 threads) so the
# proxy predicts the same congestion the real route faces and finishes fast.
CONGESTION_ESTIMATE = r"""
set_thread_count 32
set_global_routing_layer_adjustment metal2-metal10 0.3
set_routing_layers -signal metal2-metal9 -clock metal3-metal7
global_route -congestion_iterations 20 -verbose
estimate_parasitics -global_routing
report_design_area
report_wire_length
"""

# Preamble for the FINAL route in a FRESH process: read_db restores tech/LEF and
# placement but NOT liberty/sdc/wire-RC, so re-establish them. {data_dir}, {tech_dir} and
# {best_db} are filled in by the caller via str.format.
ROUTE_PRELOAD = r"""
define_corners tt
read_liberty -corner tt {tech_dir}/lib/NangateOpenCellLibrary_typical.lib
foreach _l [glob -nocomplain {data_dir}/lib/*.lib] {{
    if {{[string match "*NangateOpenCellLibrary*" $_l]}} continue
    read_liberty -corner tt $_l
}}
read_db {best_db}
read_sdc {data_dir}/constraint.sdc
set_wire_rc -signal -layer metal3
set_wire_rc -clock  -layer metal5
puts "ROUTE_PRELOAD_DONE"
"""
