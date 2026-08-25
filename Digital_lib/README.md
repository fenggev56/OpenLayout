# Digital_lib — technology files

Process/PDK files shared by every design. Design-specific macro LEF/LIB
(fakeram, custom blocks) stay in the design's `data_dir/lef` and `data_dir/lib`.

```
Digital_lib/Nangate45/lib/NangateOpenCellLibrary_typical.lib   std-cell liberty (tt)
Digital_lib/Nangate45/lef/NangateOpenCellLibrary.tech.lef      tech LEF (layers, vias, rules)
Digital_lib/Nangate45/lef/NangateOpenCellLibrary.macro.mod.lef std-cell LEF
```

Selected in `configs/<design>.yaml` via `tech.dir` (relative to the project
root or absolute). Passed to every TCL template as `tech_dir`, so
`${tech_dir}/lib` / `${tech_dir}/lef` hold the PDK and `${data_dir}/lib` /
`${data_dir}/lef` hold the design's own macros.
