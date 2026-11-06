"""THE CORE, REWRITTEN (Recalibration 2, 2026-09-06).

Layers, built and measured in order:
  entities  — components -> sprites (rigid, multi-colour) -> classes;
              identity across frames; an EVENT diff per transition
  rules     — effect templates over entities, fitted to transitions
  goals     — goal templates searched over entity relations
  planner   — search over the induced program; experiment selection
The upload gate (tools/mechanics_gate.py) scores what this core PREDICTS
about each game's level 1 against ground truth it never sees.
"""
