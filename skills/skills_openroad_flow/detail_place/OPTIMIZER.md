# Optimizer knob map — detail_place

| Problem | Knob |
|---|---|
| congestion / GPL diverges | place_density down (e.g. 0.55), routability_driven=1 |
| wirelength too high | place_density up slightly, timing_driven=1 |
| cells crowd macro edges | halo_um up (6-8) |
| setup WNS bad | run_resizer=1, timing_driven=1 |
| IR drop later | pdn_pitch down (denser stripes) |
| PDN channel errors | halo_um up; pdn_pitch up |
