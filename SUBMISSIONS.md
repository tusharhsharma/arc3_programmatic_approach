# Kaggle submission ledger — version -> contents -> hidden-set score

The public score is our only measurement on unseen games. Every submission
gets an entry BEFORE we know its score, stating what changed since the
last one, so deltas are attributable. (Local numbers are DEV; only this
column is real.)

| Kaggle ver | Bundle | Public score | What it contained / what changed |
|---|---|---|---|
| V1 | ~v4 (2026-08-27) | **0.21** | First submission. Built for max-over-runs (learn-then-speedrun); MAX_ACTIONS 1200. Archetypes at the time: docking (cn04 L1), basic permuter observation, hazard-robust perception. |
| V3 | ~v11 (2026-08-28) | **0.30** | THE STRATEGY PIVOT, ~8 sessions: competition mode decoded from harness source (ONE session per game -> single-pass efficiency is the score); Laws of Cheap Learning implemented (provisional single-vote beliefs, persistence until contradicted, free-information-first, maximal attribution per failure); aligner (sp80 11), multibody (m0r0 52), permuter L1 (lp85 15), touch-the-interactables prior; MAX_ACTIONS 4000 + wall-clock guard. |
| V4/V5 (submitted per policy when slot allowed) | v15 (2026-08-28) | — | Single capability family changed vs V3: permuter matured — whole-track shift certification (pixel-read tracks, geometric cyclic order incl. line-wraparound + chebyshev adjacency for diagonal rings, one-observation operator certification), inverse detection, per-cell contradiction, raw marker memory + fused-piece recovery, quotient search + cover semantics + content-keyed refutation, active-hypothesis machinery, anti-phantom mode gate (r11l protection; known cost: r11l L1 46->163). DEV effect: lp85 L1 15 / L2 34 / L3 38 (was L1-only). |

Rule: when a new bundle lands on the user's machine, add its row here
first; fill in the score when the leaderboard reports.
| V6 (2026-08-29) | v16 | **0.27** | Simulator-driven, generic-layer changes only: SLIDE movement concept (direction-consistent variable-magnitude votes -> move-until-blocked planning deltas; magnitude-dominance guard so step-noise never misclassifies — cn04 regression caught and fixed pre-ship); slide-aware wall attribution. DEV unchanged (lp85 15/34/38, cn04 247, sp80 11, m0r0 52, r11l 163; hazards 10/10 @95.28%). Sim: tuning 17/24 L1, holdout 9/18 — slide x collect and slide x hazard still weak (next fixes). |

| (next slot) | v17 (2026-08-29) | — | Sim-driven generic-layer round 2: slide discriminator v2 (>=2 magnitudes with real support — frequency-dominance misread a much-used lane); FREE WALL PER SLIDE (every slide endpoint donates the wall beyond it); blocked-move attribution guarded by the seen playfield (long unclassified slides marked phantom letterbox walls while the real wall went unlearned — corner livelock); slide-scoped color testimony (a slide crossing a goal condemned the goal color as floor; scoping it to classified slides only, after the broad version starved cn04's terrain learning and broke rotation detection — caught by invariants twice). Sim: tuning 19/24 L1 / 35 levels, holdout 9/18 / 18 levels. DEV unchanged. |

### V6 verdict (0.27, -0.03 vs V3): the click-tax hypothesis
Everything added v12-v16 NETTED NEGATIVE on hidden games. Prime suspect:
the permuter's phantom-button churn taxes every click/physics game
(r11l 46 -> 163 on DEV = ~12x per-game score loss, quadratic); hidden
sets are click-heavy. The permuter's lp85 L2/L3 gains cover fewer hidden
games than the tax costs. POLICY UPDATE (Tushar): no submission until a
bundle is PROJECTED to beat 0.30 — recover the click-tax first, then
converted zeros. v17 is HELD (its slide work is orthogonal but does not
address the tax).

| (superseded by v19) | v18 (2026-08-29) | — | THE CLICK-TAX RECOVERY (the projected V6-deficit driver): permuter mode now requires VERIFIED-BY-USE buttons (a second consistent click on the SAME key — real buttons sit still and get re-clicked; rope/physics click-noise never confirms) -> r11l 163 -> 46 (~12x per-game score on that class); plus structural piece registration (a marker-colored, standard-sized object on a track slot IS a goal piece — class existence no longer depends on catching one clean move). Contains all of v17. DEV: lp85 15/41/36, r11l 46, cn04 247, sp80 11, m0r0 52, hazards 10/10, sim tuning 19/24 / holdout 9/18. PROJECTION: V3 baseline (0.30) minus nothing known, plus permuter L2/L3 + slide + click-tax recovery -> expected > 0.30. |

| (superseded by v20) | v19 (2026-08-29) | — | v18 (click-tax recovery) + NEW GAME FAMILY: PEG-SOLITAIRE archetype (agent/pegs.py) — lf52 decoded empirically (click selects a peg, click the hole two cells away jumps it, jumped peg removed; win = one peg left); lattice + peg extraction, in-head DFS solver, VERIFIED-BY-USE gating (one 2-click experiment must remove a peg before the mode may engage — <=6 clicks worst case on non-peg games), atomic jump pairs (probes/ undo may not interleave). lf52: 0 -> L1 in 50 actions. Full DEV sweep: lf52 the ONLY change (18 levels, was 17); hazards 10/10; 14 tests. PROJECTION: strictly above v18's, which was already projected > 0.30. |

| (superseded by v21) | v20 (2026-08-29) | — | v19 + THE CARDINALITY LESSON (scaling audit): archetypes quantify over SETS of participants, never first-observed counts — button-count cap removed (verification-by-use is the principled gate; r11l still 46), boards = connected lattice components (lf52 L1 50 -> 25; phantom pegs gone). Six DEV wins, hazards 10/10, sim unchanged. |


| (superseded by v22) | v21 (2026-08-29) | — | v20 + SESSION-19 FOUNDATIONS (motion atomicity, goal tolerance + arrival, collectible class, lethal class, revisable identity + sticky timers, largest-mover anchor — m0r0 52 -> 45) + THE DOCKING ENDGAME REBUILT (cn04 L1 247 -> 170; was 0 after the foundations until this session): contact-is-the-goal (socket footprint frozen at pose commit; discovery never reruns mid-approach — proximity used to abort the dock at the moment of contact), ports-mate-by-overlap (navigate on the BODY: the piece's ports land exactly ON the socket's ports at the pose, so a full-piece template can never match at the goal), color-aware footprint subtraction (the piece renders on top of former socket cells while entering the cavity), success-is-a-recolor-read-at-the-source (frame pixels at expected port positions: recolor = connected, port color = sitting, bg = missed), refutation keyed to the pose generator's coordinates. Six DEV wins: lf52 25, r11l 46, lp85 15/41/36, cn04 170, sp80 11, m0r0 45; hazards 10/10 @94.86%; 14 tests; sim holdout 19/27 L1 / 33 levels (unchanged). PROJECTION: v20's (already > 0.30) plus five generic-layer concept families that fire on ANY game with movement/goals/collectibles/hazards, plus a faster cn04 — strictly above v20's. |

| (superseded by v23) | v22 (2026-08-29) | — | v21 + two identity-integrity fixes from the cn04 L2 investigation, both global: ROTATION PRESERVES THE SQUARES (a shape-op action kills the body's shape-keyed track while 3x3 fragments keep theirs, migrating avatar identity onto a port; rebinds that shrink the avatar >3x on a shape-op step are re-anchored by size/color/position continuity instead) and SWITCH-ONLY-AFTER-SUCCESS (_switch_piece may rebind identity only after a pixel-verified dock success — transient no-socket states during discovery scattered identity across remote fragments). DEV effect: cn04 L1 170 -> 105 (v20: 247); everything else byte-identical (lf52 25, r11l 46, lp85 15/41/36, sp80 11, m0r0 45; hazards 10/10; 14 tests; sim holdout 19/27 / 33). PROJECTION: above v21's — same capabilities, fewer wasted clicks, and identity now survives rotation on ANY shape-op hidden game. |

| (superseded by v24) | v23 (2026-08-30) | — | v22 + five docking robustness generalizations from the cn04 L2 campaign, all firing on ANY rotate-action hidden game: footprint-keyed pose refutation (refutations attach to the placed piece's world cells, surviving base renormalization after rotate presses — the old (k,tl) key forgot every refutation), committed-piece seed pin WITH clean-frame gate (multi-piece identity thrash no longer starves nav; the first pin attempt failed because the identity path's d>20 abort was a hidden transient filter — now explicit: arrival/refute require orientation matched + exact body cell count), click-a-cell-not-the-centroid (hollow pieces' centroids select nothing — eternal rotate/re-click loop), full-template orientation with body fallback (4-fold-symmetric bodies degenerate body-only j to 0, demanding rotations forever; rotate presses now need full-template evidence), immediate clean-arrival refutation (the engine recolors on the arrival frame, so lingering only cedes control to the generic layer which drags the piece off the pose). DEV: cn04 103 (v22: 105), all else identical (lf52 25, r11l 46, lp85 15/41/36, sp80 11, m0r0 45; hazards 10/10; 14 tests; sim holdout 19/27 / 33). cn04 L2 still open: spurious pixel-success on wrong poses (coincident-only filter attempt BROKE L1 and is reverted — the recolor geometry needs mapping before filtering). |

| (superseded by v25) | v24 (2026-08-30) | — | v23 + SUCCESS IS A NEW COLOR (the dock success predicate rebuilt on engine ground truth): connected ports recolor to a color that exists NOWHERE else in the level, so success = a color outside the commit-time palette at an expected coincident-port position, with ALL of the piece's expected ports connected (none still port-colored). Every prior confound dies by construction — socket frame, own body occlusion, and raw port color are all palette colors. Validated against the KNOWN connect recolor (color 3 observed at a real partial dock). Kills the wrong-pose spurious-success carousel that stalled multi-piece search. DEV byte-identical: cn04 103, lf52 25, r11l 46, lp85 15/41/36, sp80 11, m0r0 45; hazards 10/10; 14 tests; sim holdout 19/27 / 33. PROJECTION: at or above v23's — same wins, and the docking archetype now cannot false-complete on any hidden rotate-game. |

| V7 | v25 (2026-08-30) | **0.21** | THE BREADTH RELEASE: four zero->win game families converted in one session, all as gated verified-by-use archetypes, plus three generic-perception fixes. NEW FAMILIES: tu93 0->3 levels (EDGE-MAZE: passability lives on edge-midpoint colors, moves are a metered budget — self-anchored lattice-graph BFS with own action-map calibration + elimination); sb26 0->1 @9 actions (REFERENCE-SEQUENCE ASSEMBLY: reference strip + tray + slots + submit protocol; full-frame role scan because trays live beyond separator lines playfield detection crops); s5i5 0->1 (REMOTE ACTUATORS: no avatar — learn click->delta operators by experiment, recency-weighted hill-climb of marker-to-target distance); ft09 0->4 levels @4/7/14/16 actions (CONSTRAINT GLYPHS: lattice tiles cycle on click; embedded glyph pixels encode equal/differ constraints on 8 neighbors; per-cluster verification so decorative legend copies die individually). GENERIC FIXES (fire on any hidden game): wake-phantom crowd filter (cover/uncover pairs outsized the real body and inverted motion votes), shape-morphing mover pairing with distance cap (facing-flip bodies read as appear/vanish, votes fell to the 1px indicator), recurring-ambient event classes (fog redraw starved wall learning), gated-goal-yields-to-free-information with decay (state-match pads burned budgets forever). DEV: 29 levels across 10 families (was 19/6 at v20): lp85 15/41/39, sp80 11, cn04 103, m0r0 45, sb26 9, tu93 42/22/62, lf52 25, ar25 26, s5i5 41, r11l 46, ft09 4/7/14/16, hazards 10/10; 14 tests; sim holdout 18/27 (verified pre-existing noise via v24-snapshot replay). PROJECTION: the strongest expected jump yet — family coverage is the score's primary driver and it went from 6 to 10, with two families near human-optimal. |

| (superseded by v27) | v26 (2026-08-30) | — | v25 (submitted, awaiting score) + the LAYOUT-SWITCH SUPERVISOR (dc22 campaign, generic): when the avatar is visibly stuck and a click action exists, the missing move is often a side-panel button that REWRITES the platform layout — click static candidates round-robin (full-frame scan: buttons live beyond the playfield crop), a confirmed rewrite (field diff) invalidates ALL terrain knowledge (world_model.forget_terrain — walls learned on the old layout are lies on the new one), position-keyed sticky confirmation (button glyphs change with state), settle time before re-clicking a good layout away. Plus overlay/re86 tracking hardening (movement tracked over all big clusters so a dot-parse failure cannot fake "nothing moved"). dc22 itself remains open (needs SUPPORT-CONDITIONED MOVEMENT: passability from frame content, recorded in PLAN). DEV identical to v25 everywhere (28 levels on the 16-game battery); hazards 10/10; 14 tests; sim holdout 18/27 (noise band). PROJECTION: at or slightly above v25's — the supervisor fires on any hidden stuck-with-buttons game and cannot regress others (battery-verified). |

### V7 verdict (0.21, -0.06 vs V6, -0.09 vs V3): THE EAGER-MACHINERY TAX
The hidden set has now ranked the whole line: v11 = 0.30 > v16 = 0.27 >
v25 = 0.21. Every DEV gain since v11 has NET SUBTRACTED hidden points.
Mechanism (the r11l click-tax generalized): an archetype that misfires
on an unknown game burns tens-to-hundreds of actions, RHAE punishes
that quadratically, and any archetype CRASH trips the harness's
random-action fallback — worse than generic play. The DEV battery is
structurally blind to this: the gates were tuned on the very games the
battery tests. DESIGN LAW ADOPTED (fail-closed): no archetype may spend
unbounded actions on an unverified hypothesis, ever.

| V8 | v27 (2026-08-30) | **0.21** | v26 + THE FAIL-CLOSED GOVERNOR: every archetype (old and new — pegs 50, refassembly 24, actuators 70, glyphs 40, docking 170, carrier 60, aligner 40, multibody 80, overlay 40, edgemaze 110, permuter 90) is metered per level; overspend without a level-up = permanently silenced for the level; ANY exception inside an archetype also silences it (no more random-fallback storms). Plus: a lethal-color condemnation that blanks every goal candidate is auto-ignored (mis-blame must not blind goal-seek), layout-switch click cap halved to 20. Worst case on a totally alien hidden game is now a bounded, one-time cost per archetype instead of an unbounded bleed. ALL 28 DEV levels verified byte-identical under the governor (every conversion completes within its allowance); hazards 10/10; 14 tests; sim holdout unchanged. PROJECTION: strictly above v25's 0.21 (same capabilities, bounded downside); whether it beats 0.27/0.30 depends on how much of the v12-v25 tax was archetype misfire vs generic-layer drift — this submission is also the cleanest measurement of that split we can buy with one slot. |

### V8 verdict (0.21, unchanged from V7): THE DRIFT IS GENERIC, AND SCORING SAYS WHERE
The governor changed nothing — and RHAE's own math sharpens the diagnosis:
actions on UNFINISHED levels never hurt banked levels, so archetype
misfires on losing games cost zero. The 0.27 -> 0.21 loss can therefore
only be HIDDEN LEVELS v16 WON THAT THE v17+ GENERIC LAYER NOW LOSES
(or pre-win action inflation on won levels). The suspects are the
always-on changes: goal tolerance/arrival, atomicity, identity anchors,
lethal/collectible classes, slide v2 attribution.

| V9 | v28 (2026-09-03) | **0.21** | THE DRIFT REVERSION, round 1: (a) NEVER FAKE ARRIVAL — the v21 "within tolerance -> arrived" returned an empty plan up to a full movement step short of every goal; any hidden game whose goals must be TOUCHED silently loses levels a simpler agent had won; replaced by deterministic nearest-cell tie-breaking (the cn04 oscillation it was built for stays fixed — cn04 103 verified). (b) TWO WITNESSES TO CONDEMN a lethal color — one death near an innocent object must not strike its color from the goal set. (c) HYPOTHESIS LEDGER (agent.hlog): every adopted belief logged with the alternatives that were also consistent at adoption time (goal choices, lethal condemnations, collectible adoptions, archetype crash/overspend) — the "multiple genuine answers" record. DEV: all 28 levels identical; 14 tests; sim holdout 17/27 (-1: the one lucky maze of the already-known-weak slide-x-collect combo — accepted, the never-fake-arrival logic protects touch-goals everywhere). PROJECTION: if arrival-fakery was the drift, v28 recovers toward 0.27+; if unchanged again, next suspects are atomicity and slide-v2 attribution, and the following slot tests those. |

### V9 verdict (0.21, third identical): 0.21 IS THE GENERIC-EXPLORER FLOOR
V1 (a different agent) also scored 0.21. Three unrelated bundles landing on
the same number = the same hidden levels cleared by any decent explorer.
v11's +0.09 came from ITS ORGANS' behavior — organs we then modified. So
the study moved from guessing to MEASURING: Tushar recovered the v11
notebook (Kaggle Version 3); v11 and current now run side by side.

THE v11 STUDY (DEV, identical runner): current dominates v11 on every
game (v11: 5 hazards + lp85 L1, sp80, cn04 251, m0r0 52, r11l 46 — 13
levels; current 28) EXCEPT ONE: mz02 (hazard CURRENT) L1 68 -> 130.
Lockstep replay found the first divergent action (step 9): identical
beliefs, different PLANNER policy — v11's plan_to returned None for an
unreachable target and the agent kept probing (clean model, 68);
session-19's GOAL TOLERANCE planned to "the nearest cell" on two
drift-contaminated vectors and committed (130). Lesson: UNREACHABLE IS
INFORMATION. Two further false beliefs surfaced by the new hypothesis
ledger: water adopted as a "collectible" (mz02, step 5) and goal choice
flapping between four equally defensible colors (r11l).

| V10 | v29 (2026-09-03) | **0.24** | THE v11 STUDY RELEASE — learnings, not reversions: (1) UNREACHABLE IS INFORMATION: plan_to returns None when the exact target is unreachable (v11 semantics); tolerance survives only as an explicit opt-in for docking-class games (rotate actions) where off-lattice pose targets justified it. mz02 130->68, lp85 L2 41->34 on the strict path, cn04 103->99 (best ever), m0r0 46. (2) SLIDES AND CURRENTS BOTH VARY MAGNITUDE for different reasons: the slide discriminator now subtracts the estimated drift first (mz02's ACTION4 was a "slide"). (3) COLLECTIBLES ARE SPARSE: terrain-scale colors (>15% of the playfield) cannot be collectible classes. (4) COMMIT UNTIL REFUTED: goal choice among equally defensible candidates is sticky until the chosen one is gone/gated. (5) Hypothesis ledger in place (from v28). A permuter-gate loosening (conf 2->1) was tried and REJECTED by the data — r11l click-tax returned (46->176) — reverted; the ledger records it as a refuted hypothesis, not a tuning. DEV: 28 levels; hazards 10/10 (mz02 halved); 14 tests; sim holdout 18/27 / 33 levels (v28: 17/29 — patience helps the sim too). PROJECTION: a real chance to recover toward 0.27-0.30: the one measurable regression vs v11 on a hazard-class game is repaired at the concept level, and hazard-class rhymes are the most likely hidden overlap. |

### V10 verdict (0.24, submitted 2026-09-04): THE FIRST RECOVERY — partial hit

Timeline: V3 0.30 (v11) > V6 0.27 (v16) > V7/V8/V9 0.21 (v25/v27/v28) >
**V10 0.24 (v29)**. Prediction was "recover toward 0.27-0.30": direction
right, magnitude short — logged as a PARTIAL HIT (calibration record:
1 partial / 0 full / 0 miss under pre-registration).
WHY (from the v29 diff, one bit cannot split four changes): the
0.21 floor was three identical scores from bundles whose only common
change vs v16 was the ALWAYS-ON layer; v28 removed the arrival fake and
scored 0.21 — so arrival-fakery alone was not the drift. v29 added the
v11 planner semantics (UNREACHABLE IS INFORMATION), drift-aware slide
classification, sparse collectibles, and sticky goal choice, and the
score moved. The mechanism the study measured (mz02 130 -> 68: an
agent that plans to "the nearest cell" on a half-learned model commits
early and wastes the level; an agent that declines and keeps learning
wins it cheaply) is the most likely carrier: on hidden games with
currents/slides/drift the v12-v28 line was acting on approximations and
v29 waits for the model. HOW MUCH: +0.03 averaged over the hidden set
is one or two levels won back, or several already-won levels done
faster (RHAE is quadratic in actions) — consistent with the study's
"a few games rhyme with mz02" expectation, not with a general repair.
What remains between 0.24 and 0.30 is still generic drift from v12+
(atomicity, slide-v2 wall attribution, identity revision) — the
lockstep tool against the v11 reference is the way to find it.

| V11 | v30 (2026-09-04, submitted 2026-09-05) | **0.23** | BRANCH A's FIRST RELEASE — the atom v0 (conditioned laws; causal click laws; product-state planner; identity/goal/frontier epistemics), strictly bounded: with no law adopted the planner is byte-identical to v29; laws need >=3 witnesses a side (clicks: a distinctive body, 2 witnesses) and die on one counter-example; teleport edges bounded to scarce classes. Exploration in tiers (unpredicted cells first, then up to four class-scored clicks, then the exhaustive walk). DEV: 29 levels (v29: 28) — ls20 0->1 (27 actions), cn04 99->168 (docking trajectory fragility), m0r0 46->50, everything else action-identical (r11l 46, tu93 3, lp85 15/41/39). Sim: tuning 46->49 levels, HOLDOUT 33->34 levels / 18->19 games (never inspected). Blind set 1: sx05 0->2 levels (laws), sx06 1->2 (click law), M1 2/6. Bundle 7170 lines, validated bundle==modular on sx05/sx06/mz02/r11l/ls20/st82 and repo-free exec. PRE-REGISTERED PREDICTION: score in [v29 - 0.02, v29 + 0.04] = [0.22, 0.28] now that V10 = 0.24. Mechanism under test: exploration efficiency on open-room hidden levels (the strict frontier) and decoy-goal refutation; the laws themselves are expected to fire rarely. A miss below v29 - 0.02 would indict the exploration order (tu93-style trajectory fragility in a hidden archetype game), not the laws. Submit only AFTER v29 has its own score (one bit per slot). |

### V11 verdict (0.23): prediction HIT, at the low edge — the mechanism did not pay

Pre-registered [0.22, 0.28]; observed 0.23 (v29 - 0.01, inside rounding of
"unchanged"). Calibration record: 1 partial (V10) / 1 hit (V11) / 0 miss.
READING: the exploration-efficiency mechanism (strict frontier, decoy-goal
refutation) bought nothing measurable on the hidden set, and the atom's
first laws — as predicted — fired too rarely to move the score either
way. The one DEV level v30 gained (ls20) did not rhyme with anything
hidden. Consistent with the standing diagnosis: the core is today the
smallest part of the stack; the score is bounded by goal inference and
priors, not by exploration order. No submission until Branch A produces
a goal-induction organ with a measured DEV/sim effect.

### v31 STAGED (2026-09-05, not yet submitted — the daily slot went to v30)

| V12 (planned) | v31 | pre-registered **[0.23, 0.29]**, point 0.26 | BRANCH A's second release: THE GOAL ORGAN (agent/goals.py — first-frame relations as ranked hypotheses; effect learning over keys, buttons and relative clicks; greedy descent; selection as a mode with re-check; unblock; planner hand-off; cross-level transfer) + the single-motion fallback fix + walked-over-is-not-not-the-goal. DEV 17 -> 22 levels on 24 real games; the 8-game holdout goes 0/8 -> 2/8 L1 (vc33 17 actions, su15 10); blind set 1 L1 4/6 -> 5/6; simbench tuning unchanged (26/33, 49 levels), sim holdout 17/27 (the two losses predate this build). |

WHY THIS RANGE: the organ wins first levels on never-seen games whose
first frame shows a same-colour / socket / HUD-twin relation (2 of 8
holdout games here; the hidden set is presumably similar in kind), and
each such level is worth roughly +0.01-0.02 averaged; it costs 10-20
actions on levels an archetype already wins (ft09, lp85 L1), a small
quadratic penalty. HIT: 0.25-0.29. PARTIAL: 0.23-0.24 (the cost and
the gain cancel; hidden games do not rhyme with vc33/su15). MISS: below
0.23 (the organ's exploration loses levels v30 won) or above 0.29
(more hidden games read like these than the holdout suggested).
Submit only if Tushar agrees; one slot per UTC day.

### v32 STAGED (2026-09-21, not submitted — THE CORE as a submission agent)

| (not submitted) | v32 = submission/my_agent_core.py (2026-09-21) | pre-registered **[0.12, 0.24]**, point 0.18 | THE NEW CORE ALONE (agent/core: scene tracker, fiddling scientist, relation-language readings, laws over kinds and forms, planner composing induced laws), bundled by tools/build_core_submission.py as a virtual package (no name collisions; the adapter inverts control — the core drives a bridge environment from a worker thread, one game at a time, and the harness reads its next action). Verified through the adapter offline: tn36 L1 41, cn04 18/67 (identical to the modular runs). MEASURED BEFORE STAGING, all three honest gauges: mechanics gate 6/25 PASS (entity recall 0.71, rule recall 0.68, goals right 12/25) while it WINS 19/25 first levels — it plays what its report cannot yet say (tn36 won, reported align not inside; lf52 won, no "count"; wa30 won, reported reach); BLIND SET 2 (sealed, phase-1 only): 0/6 first levels (the old agent's generic play had cleared 4/6); SIM HOLDOUT (never inspected, counts only): 3/27 first levels (the old agent: 18-19/27). |

WHY THIS RANGE, and why it is NOT to be submitted under the standing rule:
the sealed sets say what Tushar suspected — the core wins the visible
games it was aimed at (19/25 first levels, 12 of them in the human zone)
and does not cold-start on mechanics its vocabulary lacks (0/6 blind,
3/27 sim holdout, against 4/6 and 18/27 for the naive explorer it
replaces). A hidden set that rhymes with the public games in kind pays
the 19/25; one that does not pays the 0/6. The pre-registered point is
below v11's 0.30 and below v29's 0.24. Under the gate (25/25 predicted
mechanics; two untuned visible games fallen unaimed) v32 stays staged
beside v31. What would move it: the fiddle-and-read loop coining a
reading on a blind game (M1 > 0 on set 2 with the core), and the report
saying what the play knows (count / program / carry / launch as goal
and rule words). If a slot is spent anyway, the score is the honest
measure of "aimed organs, stated generally" on unseen games — a
calibration bit worth having, at the cost of one day's slot.

### v33 STAGED (2026-09-25, ready — THE CORE OVER THE FLOOR; same day: the core watches the floor, reads and coins its wins, identity by colour and form (PLAN 36l); the slide law, a slide teaching ground and walls, the floor's clean second turn (PLAN 36m). Final sealed measure at the 400 cap: sim holdout 21/27 (7 by the core alone), blind set 2 first levels 3/6 + sx09 L2 (sx07 won at a 1200 cap); tuning sims the core alone 24/33 (21 before); lessons confirmed by replay: shf0, shf1, shw0, st22 (coined); regression 21/21 unchanged)

| (staged) | v33 = submission/my_agent_core.py (2026-09-25) | pre-registered **[0.21, 0.33]**, point 0.26 | THE JOINED AGENT: the core (v32, unchanged organs) as the layer above THE FLOOR (v11's crude loop, embedded whole), joined in one loop, `play_game`: the core reads and plans first; a level it leaves (no_plan, or a death that taught no hazard) is played by the floor from where it stands, in turns of 150 actions doubling per turn; the core gets the level back only from a state it has not tried (at most 3 tries); a level is never given up while the game runs (4000-action cap, 100-min clock). The death's second RESET (a full reset to level 1 in this engine — every death used to replay the game from the first level) is gone. In the harness the core lock is held per core turn, so the other games play their floors while one game's core thinks. MEASURED BEFORE STAGING (400-action cap, the floor's own measurement conditions; sealed sets counts only): BLIND SET 2 first levels 4/6 (sx07 08 09 10) + sx10 L2 — core alone 0/6, floor bar 4/6; SIM HOLDOUT first levels 21/27 — core alone 3/27, floor bar 18-19/27 (5 by the core, 16 by the floor); visible first levels 21/25 (core alone 19/25: sp80 and sb26 added by the floor, nothing lost — the core plays first, unchanged); tn36 L1 41 unchanged; adapter check through the harness loop: sp80 L1 won, two games in parallel threads fine. |

EVENING RE-MEASURE (PLAN 36n — the deaths class: blame reaches only as
far as the move, a lost frame teaches no law, unknown keys asked, carried
laws kept, once more from the start knowing, plan before the first
press, blind presses avoid known hazards): sim holdout 23/27 with ELEVEN
by the core alone (3 at the session's start); blind set 2 first levels
3/6 at the 400 cap, sx11 L1 and L2 by the CORE (5 and 6 actions) — the
first blind game the core solves itself; tuning sims the core alone
28/33 (21 at the day's start); two-level public regression clean
against the start-of-session snapshot.  Pre-registered range raised to
**[0.21, 0.36]**, point 0.28: the floor's 0.21 stays the lower edge; the
core now clears by itself 11/27 of a holdout it cleared 3/27 of on the
last submission's code, and one blind game outright.

WHY THIS RANGE: the floor alone earned 0.21 on the hidden set three
times (V1, V7-V9, and V3 now reads 0.21 too); the joined agent cannot
score below what the floor clears (the core's turns cost actions on
unfinished levels only, which RHAE does not charge, and the core plays
first only where it then hands over) — so 0.21 is the lower edge, less
the action inflation on levels the floor wins after the core's 16-74
probing actions. Above it: the core's own wins on hidden levels the
floor never cleared (19/25 visible first levels, 12 in the human zone;
5/27 sim-holdout first levels the floor did not need) and the sim
holdout's +2-3 over the floor's bar. The gate (6/25 PASS) is unchanged;
under the standing rule this stays staged until Tushar says the
post-mortem's rule — submit the joined agent once it beats the floor on
the sealed sets, which it now does — replaces it.

## Submission policy (agreed 2026-08-28; corrected same day)

**ONE submission per day** (verified on the Kaggle dialog — "daily
Submission allowance (1)"; resets at UTC midnight). The public score
keeps our best, so a submission can never lower the standing number —
but the single daily slot is the scarce resource.

RULES:
1. Submit only when a bundle carries clear expected hidden-set delta —
   a new capability family, or a measured DEV gain with zero
   regressions. The ledger row is written BEFORE submitting, so every
   spent slot has a stated hypothesis.
2. With 1/day, BATCH improvements: a submission should carry the day's
   full set of validated changes, not one variable. Single-variable
   attribution comes from the local ablation harness (tools/ablation.py),
   not from spending slots.
3. Never submit two identical notebook versions (V4 == V5 taught us:
   the version diff on Kaggle tells you if content changed — +0 -0
   means don't bother).
4. Kaggle privately RE-RUNS the notebook against hidden games — the
   notebook wall-clock limit applies there, which is why the bundle's
   TIME_BUDGET_S guard must stay.

### V12 VERDICT (2026-09-25/26): 0.24 — and what the scale means

| V12 | v33 (2026-09-25) | **0.24** | THE JOINED AGENT (the core over the floor, with the day's six death-class fixes). Same as V10's 0.24; V11 0.23; V9 0.21. |

THE SCALE, READ AT LAST: the leaderboard's leader stands at 19.45 with
73 entries on ~50% of the test data; our 0.24 is on the SAME column.
The official metric is the average over games of the level-weighted
RHAE in PERCENT (0–100).  So 0.21–0.24 is not "a fifth of the human
rate"; it is a quarter of one percent — one or two first levels cleared
across the whole hidden set (110 private games, per another team's
public write-up).  Every reading in this ledger that compared 0.21 with
the local RHAE of 0.044 (4.4%) had the scale wrong: on the hidden set we
clear roughly one twentieth of what we clear on the public games.

EXTERNAL EVIDENCE (github.com/BDR-Pro/arc-prize-2026-arc-agi-3, a
programmatic agent with a learned transition graph, no LLM): reports a
leaderboard of 0.26–0.27 as its "hard floor", 110 private games, and
the top of the board held by GPU LLM-agents (then ~3.57; now 19.45).
Five very different bundles of ours (V1, V9, V10, V11, V12) and their
v79 land in the same 0.21–0.27 band: that band is the programmatic
floor of this competition, not ours in particular.  The core's
contribution on the hidden set is nil at this resolution (V12 = V10).

WHAT THIS DOES NOT SAY: that the core is wrong, or that the day's work
was wasted — the sealed gauges moved (sim holdout 3→11 first levels by
the core alone; one blind game solved outright), and the confirm-by-
replay method is the paper's own result.  It says the hidden games are
of a kind neither our public set nor our sims contain, and that no
programmatic agent in the field reads them yet.

DECISION PENDING (Tushar): (a) the Paper Prize path as planned —
methodology, M1, the floor/core architecture, honest numbers; (b) an
LLM-in-the-loop variant for the leaderboard — a local open-weight model
on Kaggle's allowed GPU, given the core's readings, laws and the floor
as tools, which is how the top of the board is being played.  (b) is a
new project against a Nov 2 deadline.

### v34 STAGED (2026-09-26 — THE HARNESS FIT; PLAN 36o)

| (staged) | v34 = submission/my_agent_core.py (2026-09-26, 438 KB) | pre-registered **[0.20, 1.5]**, point 0.5 | The same agent as V12 with its timing fitted to the harness: per-thread core state, min(8, cpu) core seats instead of one lock, a 6-hour run clock shared by every game (was 100 min from a common start), core turns at a quarter of the remaining clock (≤ 600 s), plan searches at a quarter of the turn, the floor's turns capped at 600 actions and at half a level's remaining actions while the core has untried states. Dress rehearsal (eight games at once, two seats, 25 min): every game got core turns; 5 of 8 reached level 2. |

WHY THIS RANGE: V12 = V10 says the core never reached the hidden games
under the old timing; v34 gives it roughly 13 minutes per game there.
The lower edge is the floor's band less the actions the core's turns
now spend on unfinished levels; the upper edge assumes the core's
public-game first-level rate (19/25) transfers to a few hidden games
at human-zone efficiency, which one hidden reading has never shown.
The point is a guess; the submission is the measurement.  Requires the
notebook run with NO accelerator; set ARC_RUN_BUDGET_S higher only once
the Code Requirements limit is read.

### V14 VERDICT (2026-09-26): 0.24 — prediction hit at its floor; the mechanism did not run

V14 (= v34) scored 0.24, the same as V10 and V12.  Inside [0.20, 1.5]
only because the lower edge was the floor's band.  The range assumed
the core would reach the hidden games with ~13 minutes each; the
rehearsal of the true rerun path (tools/gateway_run.py, 104 games at
once over a local competition-mode gateway; PLAN 36q) shows it did not:
the eight core seats went to the first threads to reach them, every
other game "played its floor meanwhile", and the floor took every first
level at hundreds of actions (public mean 0.47 under that seating,
against ≈ 4.1 one game at a time).  0.24 is the floor alone, for five
bundles.  Verdict: prediction hit by accident of the lower edge; the
ledger's V10–V14 rows all measured the floor.

### v35 STAGED (2026-09-26 — ONE GAME AT A TIME PER SEAT; PLAN 36q)

| (staged) | v35 = submission/my_agent_core.py (2026-09-26, 441 KB) | pre-registered **[0.5, 3.5]**, point 1.3 | The v34 agent with the seating changed and nothing else: a seat is a whole game's turn at the CPU (core and floor), two seats, held from first action to last; the other games wait without acting; a seated game's clock is a fair slice of the run's remaining time (Kaggle: ≈ 490 s a game); no random play after a game's turn (its thread waits, and gives one RESET when every game is done or at the deadline — the gateway closes a scorecard idle 15 min). Hardened on the second look (2026-09-27): a game whose harness loop dies (a gateway stall past its 10 s timeout) releases its seat; a frame that does not come is asked for again three times before the game is given up; the bridge closes for good once the harness stops asking. Rehearsed on the rerun's own path: 25 public games at once over a local competition-mode gateway, 240 s slices — public mean **6.66**, 30 levels (ft09 all six), 0 faults; the same games under v34's seating: 0.47. |

WHY THIS RANGE: the hidden games are not the public ones; the public
mean under this seating is 6.66 (Kaggle's slices are twice as long as
the rehearsal's).  The Duck's public-to-hidden transfer is 0.55; ours
has never been measured with the core actually playing, and the
vocabulary was built on the public games, so the lower edge assumes a
transfer of ~0.08 (a couple of hidden first levels read at human-zone
efficiency) and the upper edge ~0.5.  Anything above 0.30 confirms
the diagnosis that the hidden set had been measuring the floor; a
fifth 0.24 would say the rerun differs from the rehearsal in a way
still unseen (the gateway's timing, the notebook's own limits).
Requires the notebook run with NO accelerator; ARC_RUN_BUDGET_S at its
default 7.5 h.

### V15 VERDICT (2026-09-27): 0.00 — outside [0.5, 3.5]; the run ended early

v35 (whole-game seats) scored 0.00.  The rerun's own path was then read
off the notebook page (PLAN 36r) and reproduced locally
(tools/kaggle_run.sh); v35 passes it, so the break is in the gateway
sidecar's behaviour that the local server lacks.  The one thing v35 did
that no earlier bundle did was leave ~108 games idle for hours before
their first action.  Verdict: miss; the seating must keep every game
acting.  The lesson for the ledger: a rehearsal that passes locally
does not confirm the rerun — only a design that gives the gateway
nothing it has not already accepted (games that act throughout, as
v34's did) is confirmed by the record.

### v36 STAGED (2026-09-27 — THE SEAT IS A TURN; PLAN 36r)

| (staged) | v36 = submission/my_agent_core.py (2026-09-27, 433 KB) | pre-registered **[0.3, 3.5]**, point 1.2 | The v35 agent with the seating changed once more: every game plays at once, one thinking at a time through one fair seat, released while the gateway answers; the core's clocks run on seat time (a game's budget ≈ 245 s of thinking on Kaggle); every game acts from the first minute to the end, as v34's did. Rehearsed on the notebook's exact path (25 public games at once, 120 s of seat time each): public mean **6.68**, 32 levels, 0 errors. |

WHY THIS RANGE: as for v35 — the rehearsal's public mean is 6.68 and
the transfer to hidden mechanics is unmeasured — with the lower edge
brought down to the floor's own band, because v36 keeps the floor
acting on every game from the start (what scored 0.24 in V10–V14) and
adds the core's turns on top.  Below 0.20 would say the sidecar rejects
something else in the bundle; the next step would then be a bundle that
differs from V14 in one line only.

### V16, V17 VERDICT (2026-09-28/29): "Kaggle Error" twice — the fault is ours

v36 and v36b (the seat as a turn; the second with the harness's frame
list trimmed and the final RESETs spread) both came back "Kaggle Error"
— the label with no traceback and no score, which Kaggle gives when the
run is lost on its side (a container killed for memory is one known
cause).  Both were 7.5 h runs of a seating no earlier bundle had; the
7.5 h runs of v33/v34 did not fail this way.  Checked here: 104 games
at once peak at 1.86 GB and grow ~165 KB an action; one game's live
state is ~150 MB after 2 000 actions; a full Kaggle run estimates at
6–16 GB against 30.  Not decisive.

### v36c STAGED (2026-09-29 — THE SAME SEATING, A TWO-HOUR RUN)

| (staged) | v36c = submission/my_agent_core.py (2026-09-29, 434 KB) | pre-registered: any "Succeeded" ≥ 0.20 says the fault grows with the run's length (memory, or a limit unseen) and the seating is sound; "Kaggle Error" again says the fault is in the seating's mechanism regardless of length | v36b with one constant changed: the run budget 2 h instead of 7.5 h (≈ 61 s of thinking a game on Kaggle; here 104 games at 23 s each scored 3.51 on the public set, so a score is still expected). |

### V18 VERDICT (2026-09-29): "Kaggle Error" at 42 minutes — the fault does not grow with the run

v36c (the 2 h budget) failed at 42 min as the 7.5 h runs did.  Not
memory, not a limit on length: something in the seating's mechanism
within the first three quarters of an hour.  Measured on the public
games over the notebook's path: with one seat and no limit on a hold,
the whole run goes silent for as long as one thinker thinks — gaps of
91 s and 167 s in ten minutes on public games; on 110 unknown games a
single unbounded think stalls every game's actions for minutes.  The
one property v36 has that v34 and v35 did not: minutes of global
silence at the gateway.

### v36d STAGED (2026-09-29 — THE SEAT ABANDONED AT 45 s)

| (staged) | v36d = submission/my_agent_core.py (2026-09-29, 434 KB) | pre-registered: "Succeeded" at any score says the silence was the fault; [0.3, 3.5] as for v36 if it scores; "Kaggle Error" again says the mechanism is rejected for a reason still unseen, and the next bundle is V14's shape with one change | v36b with the seat's hold limit 600 s → 45 s (a holder past it is left to finish on its own time while the next in line thinks), the silence watchdog 660 s → 240 s (a game whose thinker is stuck plays on without it and keeps the gateway's clock alive), the worker released when its game falls back, and the 7.5 h budget restored. Rehearsed on the notebook's path: 25 games, 60 s of seat time each — public mean 4.69, 22 levels, 0 errors, 12 holds abandoned at 45–47 s, one game to the fallback after a 4-minute think; largest silence 91 s (was 167 s). |

### V19 VERDICT (2026-09-29): 0.20 — Succeeded; the silence was the fault; the core transfers nothing

v36d ran the full course and scored 0.20.  By the pre-registered rule:
the seating is sound (the silence of one seat was what the gateway
rejected), and the number is the first measurement of the core on the
hidden games — inside the floor's band (0.21–0.24), a shade under it.
Read plainly: with every game reached and ~245 s of thinking each, the
core adds nothing the floor did not already score; its turns cost a
few actions on levels the floor might have won.  The claim of the
project as implemented — laws by directed experiment on a scene of
persistent objects win first levels of games never seen — is not
supported at this budget.  The vocabulary is fitted to the public 25.

### v37 STAGED (2026-10-05 — THE LEDGER, THE RE-TEST, THE METHOD THAT CARRIES)

| (staged) | v37 = submission/my_agent_core.py (2026-10-05, 453 KB) | v36d's seating unchanged (the only thing Kaggle has accepted) + the attempt ledger (failure reports, doubts, directed re-test), the invariance fixes (cd82 recolour, probes by the scene), the method that carries (carry law exported/seeded; agents; things known by form; a bound thing moves once per press; the scene's "whole where it was, stays"). Public: 18/18 plain first levels as before (wa30 L1 54 instead of 66), wa30 L2 won (was lost), cd82/ft09/ls20/m0r0/vc33 L2 as before. Pre-registered: the seating is the same, so "Succeeded" is expected; the score reads as v36d's did — [0.20, 0.30] says the core still transfers nothing measurable at ~245 s/game (the public gains are public gains); > 0.30 says a second level or two were won on hidden games the floor cannot win, the first transfer; < 0.20 says a core turn cost actions on floor-winnable levels more than before. |

### v38 STAGED (2026-11-01 — THE LAST SLOT; the three organs of 37ai in the bundle)

| (staged for the final slot) | v38 = submission/my_agent_core.py (2026-11-01, 630 KB; delivered as my_agent_core_v38.py) | pre-registered **[0.18, 0.40]**, point 0.24 | v37's agent and seating (v36d's, the only one Kaggle has accepted) + everything from PLAN 37ah–37ai: the no-reading class's loop statements (the keys offered; the mover walks on what it was seen on; a refusal with nothing in the way is a bound), THE RUN organ (a trace drawn and taken back is a law the plan runs; the flash is the goal — sp80 L1–L2 won), THE LEVER AND THE TWIN (a thing moved by standing on another; a rewind key and the twin that re-walks the record — g50t L1 won on the first attempt), the exchange and livery readings in the tracker, plus the carried organs of 37ac–37ag (the parallel record, conditioned outcomes, arcs, wall patches, materials, hazards). Public: quick list 18/18 unchanged since s85; climbs 40 wins as s89 + sp80 L2 and g50t L1 (44 of 183 levels, 24%). BUNDLE REPAIR: the builder's module list had omitted progress, journal, claims, agents and reader since those modules were written — the staged v37 bundle could not have loaded the current core; the list is fixed and the bundle checked through the adapter offline (sp80 reaches L3 in 200 actions). |

WHY THIS RANGE: the seating is v36d's, so "Succeeded" is expected; the
floor's band (0.20–0.24) is the lower edge less a few actions the new
protocols spend on levels the floor would win (the run probe, the twin
probe: bounded, two presses each). Above it only if a hidden game
rhymes with a run key, a lever or a rewind — the first three organs
built from human replays rather than from the public games' own
readings, and the first bundle since v33 whose core is the current
one. A fifth 0.24 says the hidden games are still of a kind the
vocabulary does not hold; the honest result either way.

REHEARSED 11-01 through the official harness loop (tools/harness_run.py, the
real agents.agent.Agent base class, four games at once, 600 s): tn36 L1 39,
bp35 L1 28, sp80 L1 10 and L2 25, g50t L1 53 — the floor taking every level
the core left; no tracebacks.

SUBMITTED 11-01/02 (UTC): the day's first slot went by mistake to a RERUN OF
VERSION 19 (the v36d-era bundle), which scored **0.22** against its 0.20 of a
month before — the same code, so the hidden rerun's own noise is about
±0.02, and every reading of 0.20–0.24 in this ledger is one band.  v38
went up as notebook Version 20 (the v38 bundle: +5751 −4027 lines against
V19) and was submitted as Version 21 on 11-02 UTC; score pending.

### V21 VERDICT (2026-11-02): 0.39 — HIT, at the top of [0.18, 0.40]; the first transfer

v38 (the three organs of 37ai, the October statements of 37ac–37ah, the
bundle repair) scored **0.39** on the hidden set, against 0.22 for the
same seating a day before (V19's rerun) — the only one of fourteen
submissions to leave the 0.20–0.24 band.  By the measured rerun noise
(±0.02) the move is real: eight times the noise.  Calibration record
under pre-registration: 4 hits (V11, V12, V19, V21) / 1 hit-by-the-edge
(V14) / 1 partial (V10) / 1 miss (V15).

WHAT IT CAN AND CANNOT SAY: the seating is V19's, so the +0.17 is the
code above the floor.  V21 carries everything since V19 — the progress
loop, the parallel record, the conditioned outcome, the drift,
materials, hazards, the no-reading statements and the three organs —
so one bit cannot name the carrier; the sealed counts, which the organs
did not move, make the October statements at least as likely as the
organs.  The leaderboard shows many low-entry accounts resting at
exactly 0.39, so 0.39 is probably a plateau (a hidden level or two that
several simple agents clear), reached by us by a different road.  In
RHAE terms +0.17 over ~110 games is one first level in the human zone,
or a second level on a game the floor had started.  Rank 3096 at the
close; leaders 75.56 / 70.24 / 58.59 (LLM harnesses).

FINAL SELECTION: tick V21 (0.39) and the V19 rerun (0.22) on the
Submissions tab before the selection deadline — neither was ticked at
the time of the screenshot.

VERDICT WORDING AMENDED 11-03 (the row above is not edited): 0.39 against
[0.18, 0.40] is inside the range only at its very top — nearly a miss on
the high side.  After V14, V15 and V19 the prior was "the reader transfers
nothing"; the final bundle transferred more than the sealed counts had
suggested, and the range was too low.  The earlier high ranges (v34 to
1.5, v35 to 3.5) were the opposite error: transfer claimed before the
public set was finished (44 of 183 levels, public mean ~6.7 — the hidden
score could never exceed it).  Lesson recorded with Tushar: completion of
the public set first, then the sealed gauge, then the leaderboard.

CORRECTIONS 11-03 (on Tushar's review; earlier rows and verdicts are not edited):
1. V19 was a MISS on its numeric range: v36d's row read "[0.3, 3.5] as for
   v36 if it scores"; it scored 0.20.  Only its qualitative test ("Succeeded"
   = the seating is sound) passed.  The V19 verdict counted it a hit; wrong.
   Calibration: 3 inside (V11, V12, V21 at the top edge) / 1 inside by the
   edge for the wrong reason (V14) / 1 partial (V10) / 2 misses (V15, V19).
2. NOISE IS NOT ±0.02.  Identical code has been scored twice on two
   occasions: V19 0.20 -> 0.22, and V3's code 0.30 -> 0.21.  Two pairs bound
   nothing; swings of 0.02-0.09 occur.  V21's +0.17 is about twice the
   largest swing seen: probably real, not established.  "Eight times the
   noise" (V21 verdict above) is withdrawn.
3. V21 IS NOT ATTRIBUTABLE: it carries five weeks of changes (+5751 -4027
   lines vs V19's bundle).  No mechanism is credited with the move.
4. THE FINAL SEALED EVALUATION WAS INCOMPLETE: on the final code only lane a
   ran, 12 of its 14 results (same win count as s86 on those 12); lanes b, c
   and the blind set were not rerun.  "The organs did not move the sealed
   counts" is withdrawn: unknown.
5. sp80 L1's 10-press win is the SECOND attempt; the first dies at 30, so the
   level costs 40 presses as RHAE counts them (human 11) — not human zone.
