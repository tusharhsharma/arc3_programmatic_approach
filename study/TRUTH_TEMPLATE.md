# <game> — ground truth (from engine source; cite file:line for every claim)

## Identity
- source file(s), line count; grid/cell size; playfield layout (HUD strips, panels).

## Controls (exactly what each available action does)
- ACTION1..ACTION7 semantics; click (ACTION6) targets that matter and what they do.

## Win condition, per level (a predicate over game state, stated exactly)
- L1: ...  (what must be true for next_level(); which objects/colors/positions)
- L2..Ln: what changes (scale, new mechanic, new participants)

## Lose conditions / hazards / budgets
- what triggers game over; step budgets / timers; deaths as reset vs level restart.

## Mechanics a first-time human must discover, in the order a human would (L1)
1. ...
(state each as a RELATION: e.g. "moving into X pushes X", "clicking Y toggles Z",
 "the avatar carries a shape; entering tile T cycles its rotation")

## What is visible on the first frame that hints at the goal (free information)
- legend/reference strips, matching colors, unique objects, counters, bars.

## Minimal winning action sequence for L1 (verified by running it), and its length
