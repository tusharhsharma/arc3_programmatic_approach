# THE LADDER, FULL TURNS (2026-10-16) — against the ladder of 10-14

Every public game climbed from level 1 until a level is lost (two attempts
a level), by the core alone, with 300 s of thinking PER LEVEL (`LEVEL_SECONDS=300
TURN_FRACTION=1.0`).  The 10-14 ladder, like every "300 s a level" run since
session 36o, had in fact given each level a quarter of the game's remaining
clock — about 75 s.  Code: snapshot s69 plus the clock override (session 37k).

| game | 10-14 (≈75 s) | 10-16 (300 s) | what changed |
| --- | --- | --- | --- |
| ft09 | 6 | 6 | — |
| tu93 | 1 | 2 | L2 won since 37e (the carry); L3 dies at 4–8 |
| cd82 | 2 | 2 | L3 no plan |
| m0r0 | 2 | 2 | L3 no plan |
| vc33 | 2 | 2 | L3 no plan |
| ka59 | 2 | 2 | L3 dies at 100 |
| ls20 | 1 | **2** | the 37c regression found by bisection and fixed (a pushed thing is seen whole) |
| s5i5 | 1 | **2** | L2 won 89, plan 26/26 — by the 37k statements, not the clock (also won at 75 s) |
| re86 | 0 | **1** | a never-won game: L1 won 28, plan 8/8 — by the 37k statements (also won at 75 s) |
| wa30 | 2 | 1 | L2 LOST since 10-15 morning (dies at 72–74) — a regression to find |
| tn36 | 1 | 1 | L2 the drawn legend (open) |
| lf52 | 1 | 1 | |
| cn04 | 1 | 1 | L2 dies at the action budget, and the core CRASHES once (TypeError '>' str/int) — bug |
| bp35 | 1 | 1 | L2 now plans and dies (3, 7) instead of "lost in 1 action" |
| lp85 | 1 | 1 | |
| r11l | 1 | 1 | |
| ar25 | 1 | 1 | |
| su15 | 1 | 1 | |
| tr87 | 1 | 1 | |
| sp80 sc25 sk48 dc22 g50t sb26 | 0 | 0 | lost at L1 within seconds (no plan) — not time walls |

**31 of 183 (17%)**, from 28.  Net: +tu93 L2, +ls20 L2, +s5i5 L2, +re86 L1,
−wa30 L2.

## What the clock says

Nothing on this ladder is a pure time wall.  Every level lost at 75 s is
lost at 300 s too, nearly all of them in seconds ("no plan"), and the two
levels that moved up (s5i5 L2, re86 L1) move up at 75 s as well.  The W2
wall ("a reading, no plan") is a wall of laws and readings, not of search
time.  The quarter-clock rule can stay for the harness; for measuring the
core it costs nothing to keep.

## Open, in order of cheapness

1. cn04 L2: the crash (one traceback), then the death at the action budget;
   the s62 → s64 bisection with full turns still to run.
2. wa30 L2: lost since 10-15 morning; the morning chain is the baseline.
3. tw01 L2 (the twist suite): two like blocks touching are read as one
   32-cell thing; the story carried from L1 names the player as mover.
4. bp35 L2: dies at 3 and 7 now — the first plan walks into the spikes.
