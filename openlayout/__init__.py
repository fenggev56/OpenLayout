"""OpenLayout — open-source multi-agent physical design flow on OpenROAD.

Merges the macro-placement optimization project (``macroflow``) with the
OpenLayout multi-agent whole-flow framework:

  stage 0  floorplan     load libs/netlist + predefined floorplan DEF
  stage 1  macro_place   sample N macro-placement candidates (macroflow),
                         evaluate each with the same simple downstream
                         scripts, select the best (skipped if no macros)
  stage 2  detail_place  PDN + global/detailed placement
  stage 3  cts           clock tree synthesis + post-CTS opt
  stage 4  route         global + detailed route + post-route opt
  stage 5  finish        IR drop check, final PPA / DRC / goal report

Each post-macro stage runs a Worker -> Debugger -> Optimizer agent loop with
config-driven budgets. Agents are driven by an LLM API (OpenAI-compatible;
local vLLM via base_url); the local system collects data, sends it with
skills to the model, and executes the returned actions.
"""

__version__ = "0.1.0"
