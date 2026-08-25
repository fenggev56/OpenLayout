# Optimizer knob map — cts

| Problem | Knob |
|---|---|
| skew too high | cluster_size down, cluster_diameter down, wire_unit down |
| too many buffers / area jump | wire_unit up, cluster_size up |
| hold violations remain | repair_hold=1, hold_margin up (0.08-0.12) |
| hold buffers fail to legalize (DPL-0036) | hold_margin down, or repair_hold=0 |
| setup got worse after CTS | root_buf stronger (CLKBUF_X3/BUF_X8), repair_setup=1 |
| weak drive on deep tree | buf_list include stronger buffers |
