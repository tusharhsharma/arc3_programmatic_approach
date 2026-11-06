# Diagnosis pass — five undiagnosed DEV games (2026-08-27)

Instrumented runs (tools/diagnose.py): 800 actions each, telemetry + frame
sheets under diag/<game>/. Two agent defects found and fixed on the spot;
all standing invariants re-verified after each change (hazards 10/10 @
87.18%, lp85+r11l level 1 each, 14 unit tests green).

## Defects found in OUR agent (fixed this session)

### 1. Event-counted color votes (world_model)
A blocked bump voted "target-cell color = wall" on EVERY bump. In dc22 two
stuck cells cast 356 of 367 wall votes for the floor color. Colors are
evidence about a CLASS of cells; one cell must not testify repeatedly.
Fix: distinct-cell voting — each (cell, label) pair counts once.
(A first attempt — "background can never be wall" — regressed mz02, where
the dominant color genuinely IS the wall. Reverted; kept the data-driven fix.)

### 2. Timeout deaths poisoned innocent cells (mapper_agent)
dc22 and cn04 kill by countdown timer (a shrinking HUD bar), not by
contact. Our escalating death-poisoning blamed the avatar's final cell —
teaching lies every ~130 actions. Fix: timer detector (uniform horizontal
run in top/bottom bands whose length strictly shrinks ≥3 times =>
_timer_active); timeout deaths skip position poisoning, keep sequence
pruning.

## Per-game diagnosis

### cn04 — "plug into socket" (BEST NEXT TARGET)
Black plug piece with light-blue prongs; white socket with matching notches.
ACTION1-4 translate (step 3), ACTION5 ROTATES the piece — our motion model
misread rotation as a (-12,-3) translation. Agent already parks the piece
adjacent to the socket. Timer deaths (fixed above).
MISSING SKILL: (a) detect that an action applies a dihedral op to the
avatar (congruence test on the avatar's own shape, machinery exists in
transform_solver); (b) pose planning — BFS over (row, col, orientation)
to reach the socket with prongs aligned. Well-defined, priors exist.
Estimated: one focused session.

### m0r0 — mirrored twin (same class as ar25)
Split screen, mirrored halves, one blue key-cell per half; the two bodies
move mirror-linked. Dual-body planning skill (already queued for ar25)
pays for TWO games. Raises that skill's priority.

### dc22 — creature assembly with timer
Left room: an animated multi-part creature (head/arm/body/legs wiggle).
Right room: two target shapes. Timer bar bottom; deaths were all timeouts.
Mechanic hypothesis: arrange/morph creature parts to match the targets.
Needs a dedicated session; the timer fix removes the harmful learning.

### sp80 — timing drop
A carrier auto-scrolls right along the top; black goal-posts stand in
water; a white gap (moving target) sits in the water line. Mechanic:
fire/drop at the moment of alignment. MISSING SKILL: event timing — hold
until an autonomous track aligns with a target column, then act. We
already detect autonomous movers; the planner just never waits for them.
Medium; well-defined.

### sk48 — sequence collection, physics-ish
Yellow arena, item stack on the right, HUD shows an item sequence
(recipe). Avatar extends a blue dashed line/rope; motion beliefs are
non-grid ((6,0), (-2,0), diagonal) — gravity or rope physics. ACTION7
does something real. Hardest of the five; needs its own session.

## Revised cheapest-win ranking (toward T2: ≥6 games with ≥1 level)
1. cn04 — rotation + pose planning (skills exist in pieces)
2. m0r0 + ar25 — one dual-body skill, two games
3. sp80 — wait-for-alignment timing skill
4. lf52 — click-pair select/place (previously diagnosed)
5. ls20 — sokoban push planning (previously diagnosed)
6. dc22, sk48, ft09 — dedicated sessions after the above
