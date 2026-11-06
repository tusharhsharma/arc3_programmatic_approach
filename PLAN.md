# ARC Prize 2026 — ARC-AGI-3 Agent: Plan, Challenges, V&V

*Status: v1 draft — 2026-08-17. Working scaffold already running locally (see §6).*

## 1. What we are optimizing (the scoring math drives everything)

The official metric is **Relative Human Action Efficiency (RHAE)**:

- Per level: `score = (human_baseline_actions / agent_actions)²`, capped at **1.15**.
- Human baseline = upper-median of *first-time* human players.
- Per game: weighted average with the **1-indexed level number as weight** (later levels count more). A game can't reach 100% unless *all* levels are completed.
- Overall: simple average across games, 0–100%.
- An "action" = any command that alters game state (including RESET after a game-over). Reasoning, tool calls, and retries are **free** — only environment interactions cost.

Strategic consequences:

1. **The penalty for wasted actions is quadratic.** Taking 2× the human's actions gives 25% on that level, not 50%. Exploration must be ruthlessly cheap.
2. **Thinking is free.** We should spend heavily on internal simulation, hypothesis generation, and planning between actions — the metric never charges for compute (until Kaggle wall-clock limits bite).
3. **The 1.15 cap kills "grind one level" strategies.** Broad completion beats deep optimization; the weighting makes *later, harder* levels the score centers.
4. **Completion is a hard gate.** A game where we clear 4/5 levels superhumanly still scores below a game where we clear 5/5 sloppily-but-reasonably.

Target to win outright: 100% (= human-efficient on every level of every game) for the $700K grand prize; realistically we aim for top-5 ($40K/$15K/...) plus milestone prizes for open-sourced solutions (next milestone: **Sept 30, 2026**).

## 2. Constraints

- **Kaggle evaluation is offline** — no internet, so no frontier API models (GPT/Claude/Gemini) in the scored run. Anything we use must run on Kaggle hardware (exact specs announced at launch; historically ~1 GPU, 12h runtime).
- **All code must be open-sourced** to be prize-eligible.
- Evaluation games are **unseen** — designed to defeat memorization. Public games are for development only; overfitting to them is the classic failure mode.
- The env API: 64×64 grid frames, 16 colors, actions `RESET, ACTION1–5` (game-specific semantics), `ACTION6(x, y)` (coordinate click — 4096-way), `ACTION7` (undo, where supported). `available_actions` is provided per frame. Games report `levels_completed` / `win_levels` and terminal states `WIN` / `GAME_OVER`.

## 3. Proposed architecture (offline-first, human-priors-first)

A modular "scientist" agent: perceive → hypothesize → simulate → act, with an explicit explore/exploit budget manager. No monolithic end-to-end net; every module is testable in isolation.

**3.1 Perception (the human vision priors, hand-built):**
Connected-component object segmentation over the 16-color grid; object attributes (color, bbox, shape hash, count); frame-diffing to detect what moved/appeared/vanished after each action; entity tracking across frames (Hungarian matching on shape/color/position); detection of static structure (walls, backgrounds) vs. dynamic entities (player avatar, movables); periodicity detection for animations that advance without input.

**3.2 Action-semantics prober:**
At each new game/level, run a *minimal, systematic* probe: try each available action once or twice from safe states, record object-level effects ("ACTION2 translates the red 2×2 object up by 1"). Use ACTION7 (undo) where available to make probing nearly free. Output: a per-game action model. This is the highest-leverage cheap step — it converts an unknown game into a known control problem in ~5–15 actions.

**3.3 World model (learned online, object-level and symbolic-first):**
Induce transition rules over objects, not pixels: movement rules, collision/blocking, containment, counters, toggles, spawn/despawn, win-flag correlates. Represent as a library of parameterized rule templates (the "physics priors" humans bring: gravity, walls stop movement, same-color things behave alike, keys open locks, filling a shape completes it). Bayesian scorekeeping over competing rule hypotheses; the model must know *what it doesn't know* (calibrated uncertainty per rule) so the planner can ask for a disambiguating experiment.

**3.4 Goal inference:**
No instructions are given, so the goal must be inferred: watch for level-transition correlates (what changed right before `levels_completed` incremented), maintain competing goal hypotheses (reach location X / make pattern Y / collect all Z / match template), and prefer plans that discriminate between hypotheses when uncertain. Death states (GAME_OVER) are negative goal evidence — log their preconditions and plan around them.

**3.5 Planner:**
Once the model is confident: A* / iterative-deepening search in the learned object-level state space for a shortest *action* path to the inferred goal (the metric charges actions, not planning time). Under model uncertainty: expectimax-lite / MCTS over rule hypotheses. Replan on surprise (any transition that contradicts the model triggers model update + replan). Prefer plans robust across surviving hypotheses.

**3.6 Meta-controller (the explore/exploit banker):**
Owns the action budget. Modes: PROBE (new level, learn actions), EXPLORE (reduce model/goal uncertainty, cheapest-experiment-first), EXPLOIT (execute plan), RECOVER (post-game-over). Uses baseline-free budget heuristics (e.g., expected human actions scale with level index) to decide when exploration is no longer paying for itself in expected level-score.

**3.7 Optional neural components (only if they earn their place):**
A small open-weights VLM/LLM (Kaggle-legal) as a *hypothesis generator* — proposing goal/rule candidates from rendered frames — with the symbolic core doing verification and planning. And/or a learned dynamics model as fallback for games where symbolic induction fails. Both gated by ablation results, not vibes.

## 4. Challenges — where I want your brainstorming

**C1. The explore/exploit knife-edge (the core problem).**
Every exploration action lowers the level score quadratically, but under-exploring means wrong models and failed levels (score 0 there). Humans amortize: their "first-time" baseline *includes* their exploration. We need our exploration to be at least as cheap as a curious human's. Open question: what's the optimal probing policy — and can undo (ACTION7) + internal replay make most exploration free?

**C2. Unknown, per-game action semantics + a 4096-way click action.**
ACTION1 might be "move up" in one game and "rotate" in another. ACTION6's coordinate space is huge; clicking randomly is hopeless. Idea: restrict click candidates to object centroids / salient cells from perception. What other structural priors on "where humans would click" should we encode?

**C3. Goal inference with zero instructions and sparse signal.**
Only signal: level counter increments, WIN/GAME_OVER. Within a level there is no reward. If our goal hypothesis is wrong we can burn the whole budget. How do we generate a rich enough goal-hypothesis space without hand-writing one per public game (which won't generalize)?

**C4. Generalization to unseen games — the death trap.**
Everything tuned on public games risks being template-matching in disguise. The eval set is explicitly built to punish memorization. Our defense is the holdout protocol (§5) and priors at the level of *physics and affordances*, not game identities. Challenge: how do we know a prior is "human-universal" rather than "public-game-specific"?

**C5. Compounding model error over long horizons.**
Plans of 30+ actions on a slightly-wrong learned model fail late and expensively. Mitigations: closed-loop execution (verify each predicted frame; replan on first surprise), conservative plans under uncertainty. Is there a principled action-budget for "verify mid-plan" checkpoints?

**C6. Hidden state, timers, and stochasticity.**
Frames can animate without input; some games may have partial observability or randomness. A deterministic-world assumption will silently corrupt rule induction. Need: a stochasticity detector (repeat an action from an identical state where undo allows) and memory over frame history.

**C7. Game-overs are expensive and their causes are invisible until hit.**
Safe-exploration problem: the first death teaches you the hazard, but RESET + replay costs actions. Can we recognize hazard motifs (spikes, enemies, timers approaching zero) from priors before dying once?

**C8. Kaggle offline + runtime limits vs. our "thinking is free" strategy.**
Wall-clock is *not* free on Kaggle. Heavy search/simulation per action must fit the runtime limit across all games. We need a compute-budget manager and profiling from day 1. Also: exact hardware is unannounced — the architecture must degrade gracefully (search depth, hypothesis count are dials).

**C9. Access & tooling friction (practical, now).**
The real public games are served via API (three.arcprize.org), which this sandbox can't reach; local test games (bt11/bt33) are trivial stand-ins. I need you to either (a) run a small download script on your machine with an ARC API key to fetch the public game files into `environment_files/`, or (b) get the sandbox network allowlist extended. Also worth checking whether the official repo publishes public games as files now.

**C10. The last mile: 100% vs. top-5.**
The grand prize needs human-parity efficiency on *every* level. Realistic plan: architect for the top-5 + open-source milestone prizes, keep the 100% path alive by making per-module upgrades cheap (swap planners, add priors) rather than betting on one hero idea.

## 5. Verification & Validation plan

**Verification — are we building the thing right:**

- *Unit tests per module*: perception (segmentation/tracking on synthetic grids with known ground truth), rule induction (feed scripted transition traces, assert recovered rules), planner (known toy MDPs with optimal action counts), meta-controller (budget accounting).
- *Golden replay tests*: recorded episodes replayed through the agent must reproduce identical decisions (determinism guard); every bug fix adds a replay regression.
- *Scoring-harness parity*: our local RHAE calculator (already built) validated against the official scorecard API on the same episodes before we trust any self-reported number.
- *Determinism & seeds*: single RNG stream, seed-logged episodes, CI job runs the full suite on every commit.
- *Fault injection*: corrupt frames, unexpected `available_actions`, mid-episode GAME_OVERs — agent must never crash or hang (a crash on Kaggle = 0 on that game).
- *Kaggle dress rehearsal*: from the first month, package and submit a dumb-but-valid agent to verify the pipeline (packaging, runtime, no-internet operation) — never debug the submission mechanics in the final week.

**Validation — is it the right thing (does it win):**

- *Holdout protocol (non-negotiable)*: split public games into DEV (tune freely) and HOLDOUT (touch only at scheduled gates, e.g. biweekly). Report both; the **generalization gap (DEV − HOLDOUT RHAE)** is our primary health metric. If the gap grows, we're memorizing.
- *Capability scorecard*: per-game tags (exploration-heavy, click-heavy, hazard-heavy, long-horizon...) tracked separately, so "score went up" decomposes into *which capability* improved.
- *Ablation discipline*: every module/prior must show its RHAE contribution on DEV and survive on HOLDOUT, or it's cut. No component ships on intuition.
- *Efficiency curves*: actions-per-level vs. human baseline, per level index — tells us whether losses come from failed levels (completion problem) or bloated action counts (efficiency problem); these need different fixes.
- *Variance*: N seeds per game; we optimize the mean but watch worst-case (competition runs once).
- *Human reference runs*: you and I play new public games cold, logging our action counts — a live sanity check on what "first-time human" efficiency looks like and where our agent's behavior diverges from ours.
- *Leaderboard reality check*: early and regular Kaggle submissions; local scores only matter if they track the leaderboard.

**Gates:**

- G1 (2 wks): agent beats random by ≥5× RHAE on DEV; harness parity confirmed; first Kaggle submission accepted.
- G2 (6 wks): completes ≥50% of DEV levels; HOLDOUT gap < 30% relative.
- G3 (Sept 30): open-source milestone submission — best current agent, cleaned and documented.
- G4+: iterate on capability scorecard toward top-5.

## 6. What's already running (this session's scaffold)

- Python 3.12 venv with the official `arc-agi` toolkit (v0.9.9, installed from source) + `arcengine`, running games **fully offline/locally**.
- `agent/runner.py` — episode runner with JSONL transition logging, per-level action accounting, RESET-after-game-over handling.
- `agent/agents.py` — `RandomAgent` and a `NoveltyAgent` (state-hashed effect statistics, optimistic exploration) as harness exercisers.
- `run.py` — plays all local games and computes the official-style RHAE (quadratic, 1.15 cap, level-weighted) from `baseline_actions` in game metadata.
- Verified end-to-end on bundled test games (bt11, bt33): frames are (1, 64, 64) int grids, `available_actions` works, level completion and scoring compute correctly. (Test games are trivial; real public games needed next — see C9.)

## 7. Immediate next steps

1. **You**: get public game files (C9) — API key + download on your machine, or network allowlist.
2. Me: perception module (segmentation, tracking, diffing) + unit tests.
3. Me: action-semantics prober + object-level rule induction on real public games.
4. Together: brainstorm C1–C4 answers; lock the DEV/HOLDOUT split the day we have the public set.
5. Me: Kaggle dress-rehearsal submission skeleton.

---

## Addendum (2026-08-17, session 3): Real games acquired; Perception v2

All 25 public games downloaded via the user's API key and running locally/offline in the sandbox. Perception survey harness (`survey.py`) renders per-game contact sheets + JSON reports; batch evaluator (`batch.py`) scores the full suite.

Game taxonomy observed: navigation-with-HUD (ls20, tu93), scrolling platformers (bp35, sk48), click-puzzles (ft09, lf52, s5i5, r11l, tn36, vc33, lp85, su15), symmetry/morph (m0r0), physics/aiming (r11l). Levels per game: 6–10.

Perception v2 mechanisms (all validated by unit tests, 14 passing):
- Playfield detection: uniform runs of NON-background color = HUD strips/letterbox/dividers; uniform background runs = playable space. Band-split at divider lines, keep largest.
- Indicator mask: pixels that change on steps where NO object moved = HUD (energy bars tick on blocked bumps). This definition cannot swallow the avatar — frequency-based masks did, twice.
- Camera-scroll compensation: global-translation estimation, world-coordinate canvas (224², origin 80) with persistent color memory of scrolled-off terrain.
- Terrain-color learning: walked-on/bumped colors are terrain, excluded from goal candidates.
- Click feedback: per-object-class (shape-signature) click statistics; classes whose clicks change the world get preferred; level-completing clicks get a strong prior.

Scores (800-action cap, seed 0): our maze fixture 2/2 levels RHAE 1.15 (optimal); real games 3/185 levels (~0.2% on completion — lf52, lp85, ar25 first levels). SDK traps documented: GameAction(id) broken (use from_id); env.step needs data= kwarg for ACTION6.

Next: per-game mechanic diagnosis loop — replay each game with frame recording, inspect where the agent stalls, grow the world-model ontology (pushables, keys/doors, click-grid semantics, aim/physics) one mechanic class at a time, re-batch after each addition. The V&V discipline (fixture + unit tests + batch regression) caught every perception bug so far; keep it.

---

## Addendum (2026-08-17, session 4): Hazard robustness — all four solved

User-identified perception hazards, each with a dedicated fixture, all now passing (10/10 levels, 95.3% RHAE on the hazard suite):

- **mz02 current/drift** (0.795): full physics decomposition — net motion = action delta + global drift; drift = componentwise median of per-action max-L1 net vectors (blocked contexts only shrink motion, so the largest sample is the free-space effect); planner simulates move-clip-drift-clip against known walls. An action that consistently does nothing under drift is inferred to have delta = −drift (the "lost left arrow" recovered by logic). Absorbing states (pinned at a wall by current) escape via deliberate RESET — the world model survives the reset, so attempt 2 plans from the start with full knowledge. Engine gotcha: RESET as the first action of a level FULL-resets the whole game (set_level zeroes _action_count); guard required.
- **mz03 projectiles** (0.613): tracker classifies autonomous movers (consistent direction regardless of action); their motion is filtered out of action attribution and goal candidates; deaths mark hazard cells; game-over resync prevents learning from respawn teleports.
- **mz04 palette shifts** (1.055): when >30% of pixels change but a color→color bijection among dominant colors explains ≥85%, it's lighting, not physics — fold the inverse into a canonical remap so all knowledge survives. Crucial detail: extend the confirmed mapping to small colors too, else a re-colored goal flip-flops each theme and gets learned as a blinking HUD indicator; and tolerate residual (an avatar moving during the shift frame must not veto detection).
- **mz05 shapeshifter** (1.150 — above human baseline): identity by spatiotemporal continuity (tracker), not appearance; motion synthesized from track displacement when exact-shape diff fails; mode-based downscaling (top-left pixel sampling made a sprite with transparent corners vanish from the logical grid entirely); indicator statistics protect a 3×3 zone around the avatar (a glittering avatar blocked at a wall statistically resembles a HUD element); blocked-verdict = "nothing moved AND all appear/vanish events are avatar-local" (glitter on a bump is not a world effect).

Real-game regression: hazard suite intact, but lf52/ar25 first levels regressed (were passing) while lp85 held — the stricter blocked/effect semantics changed click-game behavior. Total: 11/193 levels incl. fixtures. Next: diagnose lf52/ar25 click-effect attribution, then per-game mechanic ontology work resumes.

Lesson bank grew: every hazard fix came from a fixture catching a specific, nameable failure mode within seconds. 14 unit tests green throughout.

---

## Addendum (2026-08-23, session 5): Regression repair + attribution hardening

Fixed the lf52/ar25/lp85 regression chain. Root causes found and closed (hazard suite stayed 10/10 through every change):

1. **Ragged animation subframes** crashed the palette remap (object-dtype arrays) — lf52 was silently dying, recorded as 0.
2. **Wobbling game panels** (lf52's grid hovers 1px perpetually): the still-step indicator masked the ENTIRE panel — i.e., the whole game — as HUD. Replaced binary pixel masking with per-pixel color memory: a change back to a previously-seen color is routine oscillation; a change to a novel color is a real effect. Game content can now live inside animated regions.
3. **Toggle effects** flip cells between recurring colors (routine by the novelty test) — click credit now comes from object-level events (appear/vanish/move), not pixel novelty.
4. **Attribution poisoning**: single coincidental moves elected motions and avatars. Now: motion vectors need 2 consistent observations; avatars must be distinctive (≤2 instances of that appearance — a grid of 39 identical cells can never be the player); coherent global motion (≥3 objects, one vector) is scenery; clicks (ACTION6) never yield motion votes nor avatar identity; GAME_OVER frames never teach physics; full-vectors require repeated samples (one teleport glitch can't win); drift needs ≥60% modal consensus across actions.
5. **Click policy**: per-position spreading (class-level scoring clicked one representative cell forever), death feedback (the click preceding a game-over is poisoned and avoided; board-reset clears position counts), and fresh random coords in fallback clicks.
6. **Probe state machine** rewritten: budget charged per attempt, refunded on success; failed positions never retried; probe spots diversify away from the last failure line (soft preference — the hard rule deadlocked); truly inert actions exhaust cleanly.
7. **New ontology module** `transform_solver.py`: learns per-action dihedral ops (rotate/flip) from single observations, BFS-plans op sequences to make a piece congruent with a template (IoU-tolerant for textured templates), detects commit actions and move-budget exhaustion (reset escape). Wired in, awaiting a game that needs it — ar25 turned out to be something else.

Scores: hazard fixtures 10/10 (87.2% RHAE). Real games: lp85 level 1 (kept), **r11l level 1 (new** — rope-physics click game, unlocked by click improvements), 12/193 total. lf52 and ar25 diagnosed precisely but need genuinely new capabilities: ar25 is a **mirrored-twin game** (two avatars move in mirror; deaths = collision at the divider; plans must account for BOTH bodies), lf52 is **sequence-memory clicking** (click order matters; wrong next-click = death). Both prior "passes" were single lucky levels.

Next: dual-body planning (ar25 class), click-sequence induction from death patterns (lf52 class), then the remaining mechanic classes (ls20 keys/locks, tu93 budgeted paths, ft09 pattern-matching).

---

## Addendum (2026-08-24, session 6): The micromouse contract — convergence machinery

User's directive: misunderstanding is allowed (humans misread games too), but the agent must provably converge — every failed attempt must make the next strictly smarter, and learned rules must persist so later runs apply them (micromouse: run 1 maps, run 2+ speed-runs). Implemented:

1. **Escalating trajectory poisoning**: each death marks the tail of the fatal path impassable; dying again at the same final cell poisons a LONGER tail. An identical fatal plan cannot repeat indefinitely — avoidance is guaranteed to converge. (Previously observed: ar25 died 3× at the same cell with the same plan.)
2. **Click-sequence ledger**: for click games, the exact (prefix, next-click) continuation that preceded a death is banned; opener-fatal clicks are banned everywhere; too many "fatal positions" auto-reclassifies the game as order-dependent (trust prefix bans, clear position bans). Search over click orders is systematic, so the correct order is found eventually.
3. Deliberate RESET and death both restart the trajectory/sequence records; the world model, motion semantics, color knowledge, and all bans persist across attempts within the level — attempt N+1 always starts strictly better informed.

Verified: hazard suite 10/10 unchanged; **r11l level 1 now in 46 actions (was 800; stable across seeds — near-baseline efficiency)**; lp85 level 1 in ~610-640 actions (2/3 seeds; it's a grind game at the action-cap boundary). Total 12/193.

New mechanic identified by experiment: **lf52 is select-and-place** — clicking a dot rings/selects it, a second click places/moves it; the dots rearrange toward some target pattern. Needs a click-PAIR operator ontology (state = dot configuration, operator = (source, dest), goal = inferred target arrangement). ar25 still needs dual-body mirrored planning. Both queued.

Also queued from user guidance: **subtask discovery** ("buy moves"): when the goal is unreachable/budget-limited, look for interactions that change reachability or extend budget (keys, switches, refills) — plan subgoal chains, micromouse-style: understand grid -> rules of where -> rules of how.

---

## Addendum (2026-08-24, session 7): The equipped scientist — priors library architecture

User's doctrine, now the standing design: **generic solutioning is the north star, but the scientist arrives equipped** — mathematics, physics, and core skills are GIVEN, never re-derived. Three layers:

- **Layer 0 — Priors library (given):** geometry (connectivity, congruence, dihedral group, symmetry, counting), physics (persistence, solidity, blocking, inertia), affordance archetypes (agent, obstacle, collectible, hazard, template/canvas, palette/tool, button), game grammar (levels, goals, deaths, HUD conventions).
- **Layer 1 — The scientist (generic):** static LOOK pass -> hypothesis generation from priors -> COMMITMENT (pursue one hypothesis to completion, never interleave competitors) -> honest refutation tests -> next hypothesis. Every experiment strictly shrinks the hypothesis space.
- **Layer 2 — Skills (cached solvers):** flood-fill, dihedral BFS, sequence search — invoked by hypotheses, not hard-coded per game.

Built this session (`agent/scene_analysis.py` + mapper integration), all as REUSABLE priors:
block-grid abstraction (a lattice of blocks = a matrix; area-weighted modal size so marker specks can't outvote cells; empty slots stay clickable), panel clustering, template/canvas match tasks, palette-compatibility (a template whose colors the canvas can't display is symbolic, not literal), symbolic bijection enumeration (mini-map legends), cellwise-consensus targets (repeated information = signal), ARC-style analogy (A:B :: C:D, reading order), select-then-paint tool tracking (brush color learned from painted result), and hypothesis COMMITMENT with honest refutation (non-responding canvas refuted in one click).

Verified: zero regressions (hazards 10/10, r11l 46 acts, lp85 638). ft09 case study: the scientist correctly refuted every naive hypothesis (all 4 patterns painted to match, + submit-click, + marker-decode — none advance the level); its true rule needs deeper experiment design — a good benchmark for the Layer-1 engine's next iteration (auto-generated experiment plans, not hand-enumerated candidates).

Standing risks tracked: holdout split still to enact; official action-counting across RESETs unverified; ls20-class joint-state (push) planning and ar25 dual-body planning queued.

---

## Addendum (2026-08-27, session 8): Two-run architecture built — the score multiplier is real

Probes #1/#2 against the live API verified: within a run every action counts
(quadratic), across runs only the BEST counts (max-over-runs, confirmed in the
scoring library). Built the two-phase architecture the same day:

- `agent/speedrun.py`: PlaybookRecorder (keeps only the final successful
  attempt per level — deaths/resets wipe the segment) + SpeedRunAgent
  (pruned replay of no-effect-free sequences; graceful degradation to a live,
  warm-started MapperAgent on any divergence; failed speed-runs are free
  under max-scoring).
- `speedrun_harness.py`: local two-phase evaluation with best-run scoring
  mirroring the official rule.

Results: fixture suite best-run 112.25% (cap 115): mz02 82→17 actions,
mz03 68→28, both at cap. **lp85 level 1: 638 learning actions → 5-action
speed-run (per-level score capped)** — the headline demonstration that
exploration is free and distilled solutions score near-perfect. r11l replay
46→46 (its physics clicks all "mattered"; optimizing that needs model-based
replanning — queued). Zero regressions (14 tests, hazards 10/10).

Next: per-level model-based replanning in speed-runs (beat verbatim pruning on
navigation games), multi-speedrun iteration (retry free, max only rises), then
the T2 mechanic ontologies — each new completed level now scores double.

---

## Addendum (2026-08-27, session 9): T0 closed — submission-ready

Identity verified (user). Built the full submission pipeline:
- `tools/build_submission.py`: bundles all 8 agent modules into the single
  `my_agent.py` the official Kaggle starter kit requires, with a top-level
  NAME-COLLISION GUARD (a duplicate `unit()` across tracker/world_model
  silently broke drift handling in the bundle — caught by fixture-validating
  the bundle itself, now impossible to ship).
- `MyAgent` framework adapter (choose_action/is_done contract, GAME_OVER
  resync, level-up detection, never-crash fallback).
- Bundle validated behaviorally identical to the modular agent on all 5
  hazard fixtures (2/2 levels each).
- Repo hygiene: MIT LICENSE, README with reproduce steps, holdout enforcement
  in batch.py (8 games skipped by default; --include-holdout for the one-shot
  T5 eval).
- SUBMIT_STEPS.md + my_agent.py placed on the user's machine; first Kaggle
  submission is now a 15-minute user action via the official starter kit
  (make setup / play-local / submit).

T0 complete. Next: T1 ablation harness, and the submission-agent question to
answer empirically from the first leaderboard run: whether competition Phase B
permits multiple runs per game (max-over-runs) — decides if MyAgent should
embed the two-phase learn-then-speedrun behavior in-run.

---

## Session 13 addendum (2026-08-27) — Kaggle T0 executed + diagnosis pass

T0 EXECUTED: first Kaggle submission pushed and submitted by Tushar
(kernel tusharhsharma/arc-prize-2026-arc-agi-3-starter, Version 1,
submission.parquet). Paper Prize eligibility established; leaderboard
result pending at session close. Windows toolchain: 4 .bat scripts
replace the Makefile (KAGGLE_1_SETUP/2_PLAY_LOCAL/3_SUBMIT/4_STATUS in
arc3-agent\submission\); fixes en route: cp1252→UTF-8 (PYTHONUTF8=1),
vendor-pull conflict, batch-parenthesis bug, and a bundle fix — removed
`from __future__ import annotations` (starter loader execs my_agent.py
without sys.modules registration; string annotations crash dataclasses).
Bundle re-validated: fixtures identical, loader-repro passes, 14 tests.

DIAGNOSIS PASS (sk48, cn04, dc22, m0r0, sp80): see DIAGNOSIS-2026-08-27.md.
Two agent defects fixed with invariants held (hazards 10/10 @87.18%,
lp85/r11l intact): (1) distinct-cell color voting — one stuck corner had
cast 356/367 wall votes in dc22; (2) timer detection — dc22/cn04 kill by
countdown, timeout deaths no longer poison positions.
New skill queue (cheapest-win order): cn04 rotation+pose planning;
m0r0+ar25 dual-body (one skill, two games); sp80 wait-for-alignment
timing; then lf52 pairs / ls20 push. Playbook: tools/diagnose.py renders
telemetry + contact sheets for any game.

---

## Session 13b addendum (2026-08-27) — cn04 SOLVED (level 1): dock-and-orient skill

NEW SKILL: agent/docking.py + mapper integration — the first T2 capability.
Trigger: a discovered rotate action (dihedral op on the avatar's own cells).
Mechanism: RIGID GROUPS (avatar body + attached parts; socket + notches);
PORT COLORS = minority colors shared across groups; pose search = rotated
placements where avatar port cells EXACTLY coincide with same-color socket
ports (engine-verified win condition), on the movement lattice, in bounds,
all full-cover candidates kept (renders can alias two port colors);
execution = rotate presses + greedy top-left navigation + SELECTION BELIEF
(select-then-act games: clicking toggles selection; click only when a
within-bounds command was ignored); refutation ledger over poses.
Supporting fixes: timer bars detected in BOTH directions (shrinking bar or
its growing complement); timer-colored appear/vanish filtered from
attribution; shape-op detection purges stale translation votes and resolves
the probe; probes no longer refund forever in timer games.
RESULTS: cn04 level 1 in 54 actions (scripted optimum 14). Real games with
>=1 level: lp85, r11l, cn04 (3 of T2 target 6). Full DEV sweep 19.92%
overall (13/136 levels; fixtures 10/10 87.18% unchanged; lp85/r11l intact;
14 tests green). Bundle v5 (2908 lines) validated WITHOUT repo on path
(import-hoisting bug found: function-local `from agent.X import` would have
crashed on Kaggle silently into random fallback — now top-level relative,
stripped by bundler; OPS alias unified).
Next: cn04 level 2+ = MULTI-PIECE docking (select next piece after one
docks); then dual-body (ar25+m0r0).

---

## Session 13c addendum (2026-08-27) — multi-piece groundwork; L2 still open

Toward cn04 level 2+ (four-piece assembly, chained port connections):
LANDED (validated, invariants green): co-movement ledger (tracks that
displace identically in one step are one rigid body — geometric adjacency
lies when pieces touch); motion-consistent grouping with NEGATIVE-evidence
exclusion + membership hysteresis (accepted parts stay); track-wide
shape-op detection with dominance guard (rotation learnable even when the
tracked part is rotation-invariant; mz05-style shapeshifters can't false-
positive); piece-switch machinery (_switch_piece) + partial-success
detection (connected ports recolor => progress, not refutation); dock
state cleared on death; 2-strike orientation tolerance; sig-keyed centroid
memory (seed switches must not read as motion).
PARKED: selection-livery seeding (learn the body-recolor of the selected
piece and seed identity from it) — mislearns when control changes pieces;
needs stricter evidence (color that appears IFF the clicked piece starts
responding). This is the key to L2: in multi-sprite select games the
avatar MUST equal the engine's selected sprite.
STATE: cn04 L1 123 actions deterministic across 3 seeds (best observed 54
on the leaner pre-multi-piece config); L2 open. Fixtures 10/10 @87.18%,
lp85/r11l intact, 14 tests green. Bundle v6 (3061 lines) validated
loader-clean without repo on path; committed to user machine.
Bisect discipline note: three regressions this session were caught by
3-seed determinism checks + engine-truth tracing; keep both in the loop.

---

## Session 13d addendum (2026-08-27) — m0r0 SOLVED (level 1): multi-body skill

FOURTH real game with a completed level. New skill: agent/multibody.py —
joint-state planning for games where several bodies answer the same input
with different (mirrored) motions. Ground truth from m0r0 engine source:
4 mirror pieces (TL:+dx+dy, TR:-dx+dy, BL:+dx-dy, BR:-dx-dy), walls block
each body independently (THE mechanism for changing relative positions),
hazards kill, win = all pieces merged (same cell; adjacent-swap also
merges). Skill: detect >=2 controlled non-autonomous tracks with
established per-action direction maps that DIFFER (dominance judged among
NONZERO displacements only — wall-blocked no-ops are wall evidence, not
direction evidence); A* over the joint state (blocked body stays); merge
archetype goal (all pairwise distances <= one step) with refutation;
prediction verification per command (a body that failed to move as
predicted marks an unknown wall — every step teaches); body memory across
death-resets keyed (color, size, spawn quadrant); planner cooldown after
failed searches (ar25 was burning 288s/run in hopeless A*; now 52s).
Builder guard added: function-local package imports now fail the build.
RESULTS: m0r0 L1 330-342 actions across 3 seeds. Real games with >=1
level: lp85, r11l, cn04, m0r0 = 4 of T2 target 6. Fixtures 10/10 @90.11%
(improved). Bundle v7 (3291 lines) validated repo-free; 14 tests green.
ar25 decoded from source (mirror-REFLECTION coverage puzzle with handle
selection + undo — different class, own session).

---

## Session 13e addendum (2026-08-27) — THE LAWS OF CHEAP LEARNING (paper material)

Question posed (Tushar): human ~30 moves/level, our speed-run 5, our live
first pass 100s — what GENERIC LAW reduces first-pass learning cost?
Answer, now named, implemented, and measured:

LAW 1 — VERIFICATION BY USE, NOT REHEARSAL. A single UNCONTESTED
observation is provisionally true; planning with it makes every purposeful
step a free confirmation trial. Only a CONTESTED belief (conflicting
outcomes observed) demands repeated evidence. Implementation:
MotionBelief.best() accepts an uncontested single vote; probe resolution
accepts one axis-aligned vote (a diagonal/large first vote is suspicious —
rotations masquerade as translations — and keeps its probe alive).
The attribution filters built earlier (global-motion, autonomous movers,
timers, avatar consistency) are what make provisional trust safe: the
2-vote rule was paying for dirty attribution; clean attribution makes
repetition redundant.

LAW 2 — PERSISTENCE UNTIL CONTRADICTED. Carried knowledge (motions,
shape-ops, inert actions) is exempt from re-probing on new levels; its
first planned use is its test, and blocked-verdict machinery catches any
change for free.

LAW 3 — FREE INFORMATION FIRST (the LOOK phase: everything inferable from
the static frame costs 0 actions). LAW 4 — MAXIMAL ATTRIBUTION PER
FAILURE (every death/block yields its full information content).

MEASURED EFFECT (all at seed 0, same caps):
- m0r0 L1: 342 -> 43 actions (8x; human-baseline zone)
- hazard fixtures: 90.11% -> 95.28% RHAE (10/10 held)
- r11l 46, lp85 638 (unchanged — their cost is elsewhere: clicks)
- cn04 L1: 123 -> 251 REGRESSION — the honest boundary of Law 1: cheap
  probing was accidentally feeding rotation detection with repetitions;
  with probes resolved early, cn04's cost shifts to selection-toggle
  churn. Cure = the parked selection-livery mechanism, not more probing.
Also: stray-click gate in rotate-games (clicks toggle selection there;
only docking machinery may click). Bundle v8 (3315 lines) validated
repo-free. 14 tests green (test updated to the new evidence contract).

---

## Session 14 addendum (2026-08-28) — lp85 L1: 638 -> 15 ACTIONS (the permuter)

THE LAWS' PUREST DEMONSTRATION. New skill agent/permuter.py: permutation-
operator puzzles (lp85 engine ground truth: buttons apply cyclic shifts of
pieces over a loop; win = tagged pieces on their goal markers; L1 allows
only 13 clicks).
Skill: (a) OP OBSERVATION — a click that rearranges several objects is a
button; correspondence built per (color,size) class by greedy nearest
bijection between prev/cur frames (loop shifts out-jump the tracker's
radius); operators MERGE across states, contradiction wipes for relearn;
button identity = SMALLEST non-bg object under the click. (b) GOAL — a
marker often WEARS THE COLOR of its piece class (color-identity assignment
first; classes without markers are unconstrained filler); goal rings of
corner fragments are clustered and their centroid snapped to the slot.
(c) SOLVE IN THE HEAD — BFS over configurations using learned operators:
zero game actions; then execute the click sequence. Cooldown throttle
caps wall-clock on unsolvable states.
RESULT: lp85 L1 in 15 actions (2 learning clicks + 13-click solution;
human baseline ~30; prior agent 638; speed-run replay 5). Deterministic
seeds 0-2. Fixtures 10/10 @95.28%; cn04 251 / m0r0 52 / r11l 46 intact;
14 tests green. Bundle v9 (3589 lines) validated; builder now also
refuses ALIASED package imports (3rd bundling bug class caught forever).
FIXED EN ROUTE: a mis-anchored state patch had been WIPING multibody +
permuter state on every observe (also explains m0r0 body-memory always 0).
lp85 L2+: bigger multi-loop instances — solver present, search budget to
tune. Next: livery for cn04 L2, sp80 timing, ar25 reflection-coverage.

---

## Session 14b addendum (2026-08-28) — sp80 SOLVED (level 1, 11 actions): the aligner

FIFTH real game with a completed level — and it OPENED in the human-
beating zone (11 actions vs ~30 human baseline; scripted optimum 4;
previous agent: zero levels ever on sp80).
Ground truth (sp80 engine): NOT a timing game as the black-box diagnosis
guessed — an arrange-then-commit puzzle: a movable DEFLECTOR platform, a
water SPOUT, two BASINS, and a limited-budget POUR (ACTION5, 4 uses).
Water hits the deflector, runs to its ends, falls just outside them; win
= both basins filled.
New skill agent/aligner.py + mapper _align_step:
- COMMIT ACTION = the ODD ONE OUT: available but neither a movement, nor
  a rotation, nor a click (cross-frame attribution of the pour's effects
  is unreliable; the odd-one-out rule is free information).
- GEOMETRY IS THE SOLVER (Law 3): winning placements computed from
  statics alone — piece spans the source column, piece ends ±1 land in
  two distinct basins; candidates ordered nearest-first; each death
  refutes one placement.
- ENGINE GRID UNITS: cell size inferred from the movement step (raw-pixel
  sizes misled every threshold until converted).
RESULTS: sp80 L1 = 11 actions, deterministic across seeds. All previous
wins intact (lp85 15, m0r0 52, r11l 46, cn04 251), fixtures 10/10
@95.28%, 14 tests green. Bundle v10 (3741 lines) validated; the ALIAS
GUARD caught its first real bug during this build.
Solved-game scorecard (L1 first-pass actions): lp85 15 · sp80 11 ·
m0r0 52 · r11l 46 · cn04 251. Five of T2 target six.
Next: lf52 or ls20 for game #6; cn04 livery; lp85 L2 search budget.

---

## Session 14c addendum (2026-08-28) — ls20 opened (carrier-matching, WIP)

ls20 decoded from engine source: NOT sokoban — an ATTRIBUTE-CYCLING
COLLECTOR: the avatar carries a displayed shape; stepping onto special
tiles cycles its FORM / COLOR / ROTATION (one cycle per tile ENTRY —
step off and back on to cycle again); a gate displays the REQUIRED shape
and blocks until the carried shape matches; step budget with lives.
L1 concretely: one rotation tile, one gate; scripted probe confirmed
rot 270->0 on first entry, match=True, +90 deg per re-entry.

Carrier-matching skill (agent/mapper_agent.py _carrier_step + tile-effect
learning in observe):
- TILE-EFFECT LEARNING (Law 1): avatar steps onto a tile class and exactly
  ONE far, SIZE-PRESERVING display change occurs -> that class's attribute
  learned (rot via dihedral test / color / form); ambiguous steps skipped.
- Carried display elected by CHANGE-FREQUENCY VOTING; required display =
  most similar-sized static shape; difference chooses the tile to visit;
  matched -> walk to the gate.
STATUS: WIP — tile learning verified correct on scripted paths
(tile 9 -> 'rot'), but full-run learning parity + carried-box election
still misfire (box mislocates onto large regions). GATED: carrier is
inert in rotate-ACTION games (it hijacked cn04 until gated; cn04 restored
to 251). All five wins + fixtures re-verified after gating; 14 tests.
Bundle v11 (3899 lines) validated with carrier inert-by-default.
NEXT for ls20: robust carried-box election (persistent bbox across >=3
changes, small-area prior), then attribute-diff loop; expected win ~25-40
actions on L1.

---

## Session 14d addendum (2026-08-28) — ls20 WIP progress + generic exploration prior

Answered Tushar's generalization question in-chat (what transfers: the
full perception/model/laws stack runs on EVERY game; per-game work yields
trigger-gated ARCHETYPE skills; the hidden-set score is the referendum —
submission of v11-era bundle made today).

NEW GENERIC PRIOR (landed, all-games): TOUCH THE INTERACTABLES — a small,
scarce object class on walkable ground earns one visit (lattice-snapped
to the avatar's movement grid) before distant frontiers. Born from ls20:
the agent walked PAST its rotation tile for 400 actions.

ls20 pipeline pieces verified individually: tile-effect learning via
prev-frame lookup (avatar covers the tile it stands on), display election
with region-size cap + >=2-vote uniqueness, interior-only comparison
(displays perceive MERGED with their frames; strip 2px border ring).
Full-run composition still misfires (one false 'rot' attribution without
tile entry; carrier engaged only 6/600 acts). PARKED for a dedicated
session with a replay harness. Carrier remains inert-by-default; the
five wins + fixtures re-verified green (cn04 247, rest unchanged);
bundle v12 validated.

---

## Session 14e addendum (2026-08-28) — the northstar audit + quotient search

Tushar's challenge: "if we truly learned the concept, L2/L3 should follow."
Audit answered in-chat; the experiment run on lp85 L2 with ZERO new game
knowledge produced three permanent conceptual upgrades to the permuter:
1. QUOTIENT SEARCH: operators are pure cell-permutations => pieces never
   interact => track ONLY constrained pieces. State space collapses from
   |cells|^|pieces| to |cells|^|constrained| — tractable for ANY number of
   interlocked loops (L1 solver work now trivial; L2 search no longer the
   blocker).
2. COVER SEMANTICS UNDER PARTIAL OBSERVABILITY: goals hidden beneath
   never-moved pieces are invisible; only KNOWN goal cells constrain (a
   piece parked on its hidden goal is satisfied in place; vacating reveals
   it and grows the memory).
3. CONTENT-KEYED REFUTATION: an assignment refuted under yesterday's
   knowledge is retried when new goal cells appear (index-keyed bans were
   epistemically wrong).
lp85 L2 STILL OPEN — the residual blocker is now crisply identified and it
is a CONCEPT, not scale: ACTIVE SENSING ("act to reveal hidden goals" —
choose clicks that vacate never-observed slots). That is scientist-v2
experiment-design territory, next on the roadmap. L1 unchanged at 15;
fixtures 10/10 @95.28%; sp80/m0r0 spot-checked in bundle v13; 14 tests.

---

## Session 15 addendum (2026-08-28) — lp85 L2 SOLVED: whole-track shift certification

The "active sensing hang" from 14e dissolved under instrumentation — the
timeout was ~2.4s of solver work per action because EVERY solution was
abandoned after its first click (op re-observation wiped the pending plan
even when it merely confirmed). Fixing that exposed a chain of six deeper
defects, each producing a permanent generic upgrade:

1. PLAN SURVIVES CONFIRMATION: operator growth (new cells observed) no
   longer kills the executing solution; only contradiction does. (The
   livelock: solve 2.4s -> one click -> wipe -> repeat.)
2. FREE INFORMATION FIRST, BUT ONLY WHEN NEEDED: unlearned/open buttons
   are clicked to completion — but only after a solve has FAILED (L1
   solves with half-learned ops in 15 actions; completing first cost 12).
3. PER-CELL CONTRADICTION: a contradicted mapping deletes ONE cell's
   belief, not the whole operator (wiping erased 20 good mappings per bad
   one).
4. MARKER MEMORY = RAW OBSERVATIONS: goal-marker centroids are stored
   unsnapped and interpreted at USE time with current knowledge — snapping
   at store time froze early mistakes (fragments of unlearned loops
   snapped onto the wrong loop and haunted the memory forever). Cluster
   radius scales with the ring step (+1 rounding slack); marker material
   must be SUB-PIECE sized (piece-merges of adjacent tiles invented
   goals). A piece FUSED with its own-color marker dots is recovered as a
   piece (it vanishes from the class scan exactly when satisfied).
5. INVERSE DETECTION (Law 3): a button pair whose partial maps agree as
   inverses on >=2 cells with zero contradictions donates its mirror
   mappings — half of every operator comes free.
6. WHOLE-TRACK SHIFT CERTIFICATION (the decisive concept): per-piece
   correspondence is UNRECOVERABLE when same-color pieces ride adjacent
   slots (the greedy bijection invents swaps; every click contradicted
   the last — ops were garbage full of 2-cycles). The right observable:
   the button's TRACK (cells its clicks change, read from PIXELS over the
   piece lattice — immune to object merging), whose cyclic order is pure
   geometry (walk the unique 2-regular graph; triangle-prune corner
   shortcuts; a path = a line with WRAPAROUND, the wrap edge inferred
   from topology since no local observation can certify a teleport).
   One before/after occupancy-vector comparison against the two rotations
   certifies the ENTIRE operator in a single observation. Certified ops
   outrank fragments and stop completion clicking.
7. MODE GATE (what counts as a button game): entries that contradict
   themselves are click-noise; >8 uncertified "buttons" is not a button
   game (the permuter had hijacked r11l's rope for 280/400 actions).

Ground truth read from the engine (diagnosis only): lp85 L2 = one 26-cell
rectangle ring + two 10-cell LINES with wraparound shift, sharing cells
with the ring; 42 tiles on 42 slots (always full — cover semantics never
reveals anything by vacating; certification is the only path in).

RESULTS: lp85 L1 15 + **L2 34 actions** (human baseline ~30/level; was
unsolved at any budget). All invariants green: hazards 10/10 @ 95.28%,
sp80 11, m0r0 52, cn04 247, r11l L1 163 (slower than 46 — phantom-button
churn before the mode gate trips; known, acceptable), 14 unit tests.
Bundle v14 (4481 lines) validated repo-free, behavior-identical (15+34).
Bundler hardened: multi-line package imports now refuse to build.

L3 SOLVED THE SAME SESSION — **38 actions** — with ONE metric change:
track adjacency in CHEBYSHEV distance (an octagon's diagonal step is two
straight steps in manhattan and the walk shattered; under chebyshev every
ring step is one pitch), while the triangle prune stays MANHATTAN-strict
(under chebyshev a rectangle's corner shortcut ties its genuine legs).
lp85 now L1 15 + L2 34 + L3 38 — all three in the human-baseline zone.
Nothing else needed: markers, cover semantics, quotient search, and
certification carried unchanged. Bundle v15 (4487 lines) validated
repo-free, behavior-identical (15/34/38); committed to the user machine.

L4 OPEN (57x57, FOUR rings + FOUR lines = 16 buttons): the anti-phantom
mode gate (>8 uncertified buttons = not a button game) blocks legitimate
play there — it needs a better discriminator than count (spatial
scatter of button keys vs the r11l failure signature: many same-color
keys clustered on the piece lattice). First item next session.

---

## Session 15b addendum (2026-08-28) — T1 ablation harness + the cold-start number

Submission policy agreed (2 slots/day; spend one only for expected
hidden-set delta; ledger row written before submitting —
SUBMISSIONS.md). v15 queued for submission under that rule.

T1 ABLATION HARNESS BUILT (tools/ablation.py): MapperAgent now takes
`ablate={...}` with 8 switches (docking, carrier, aligner, multibody,
transform, permuter, blockmatch, touch); configs `full`, `naked`
(all archetypes off = the generic layer alone), and a leave-one-out
sweep. 14 tests green; competition builds ablate nothing.

THE COLD-START NUMBER (800 actions, seed 0, DEV):
- full agent, real games: 7 levels (lp85 3, cn04/m0r0/sp80/r11l 1 each)
- NAKED scientist, real games: **2 levels** (lp85 L1 via generic click
  grinding, r11l L1) — the generic layer's cold-start yield.
- 15 of 22 DEV games are at ZERO under BOTH configs.
- Fixture means mislead (mz suite ~95-115% dominates mean RHAE:
  22.67% vs 21.70%); levels-on-real-games is the honest metric.

STRATEGIC READING (Tushar's "if we really solved it we would not be at
<1% of human" — confirmed quantitatively): hidden-set score is dominated
by the ZEROS. Roadmap weighting: (1) convert zero-games family by
family — each is worth more than deepening any existing win; (2) the
generic layer's measurable target is "L1s cleared with no matching
archetype" (now trackable per run); (3) submissions spent when zeros
convert. Next conversion targets (from DIAGNOSIS + engine reads):
lf52 (select-and-place pairs), ls20 (carrier composition), ar25+
(reflection-coverage), sp80 L2+, dc22/sk48/ft09 diagnosis.

---

## Session 16 addendum (2026-08-29) — the concept-composition simulator (v0, live)

Tushar's direction (standing): the simulator runs PERMANENTLY IN
PARALLEL — a living artifact updated whenever a real game teaches a new
concept; DEV-zero conversions continue; ONE Kaggle submission/day
(corrected — see SUBMISSIONS.md policy), spent on consolidated, ledgered
bundles. Twin deliverables: winning agent + publishable paper.
Design doc: claude/concept-simulator-design.md (why it's sound, the
three failure modes designed against: generator leakage, Goodhart,
missing-prior blindness).

BUILT (v0): tools/simgen.py — composes control mapping (identity /
rotated / swapped) x movement rule (step / slide / drift) x goal
(reach / collect-3) x hazard (none / static / 40-action timer) into
arcengine games under environment_files/sg*; 6 TUNING compositions
(inspect freely) + 6 HOLDOUT compositions (unseen pairings of the same
concepts — the generalization probe; never debugged against).
tools/simbench.py reports actions-to-first-clear split tuning/holdout —
the skill-acquisition-efficiency metric (the fluid-intelligence proxy).
sg* games excluded from batch.py/ablation.py DEV sweeps.

FIRST MEASUREMENT (full agent, 400-action cap):
- tuning 4/6 games clear L1, mean first-clear 24.5 actions (human zone);
- HOLDOUT 1/6 (sg51 swapped-controls x collect3 x timer cleared in 28 —
  control remapping COMPOSES, because motion semantics are learned,
  never assumed).
- TWO CONCEPT GAPS FOUND IN MINUTES: (1) SLIDE movement (ice physics —
  multi-cell displacement per press; never built); (2) DRIFT in open
  random rooms fails although the mz02 drift fixture passes — our drift
  handling was OVERFIT to the fixture's layouts. Both now have standing
  reproductions (sg03/sg06). The honest composition-gap number comes
  after these concept fixes; today's holdout deficit is mostly the same
  two gaps recombined.

Invariants after all edits: 14 tests, hazards 10/10 @95.28%.
Queue: fix slide + drift-generalization (sim repros in hand) -> re-run
simbench -> then lf52 conversion; add new concepts to the generator as
conversions land (permuter/button axis is v1's big addition).

---

## Session 16b addendum (2026-08-29) — slide concept + simulator hardening; bundle v16

Simulator matured in three forced steps, each a methodology lesson:
1. SOLVABILITY MUST BE PROVEN UNDER COMPOSED DYNAMICS (generator now
   BFS-solves each maze with the true rules; plain-step reachability had
   certified drift-impossible and slide-unwinnable tasks — several
   "agent failures" were the simulator's fault).
2. SINGLE-INSTANCE RESULTS ARE MAZE-LUCK: 3 mazes per combo, per-combo
   clear rates (42 games; 4-char game ids — the engine loader requires).
3. TUNING MUST COVER EVERY CONCEPT PAIR WE INTEND TO FIX (slide x hazard
   / slide x collect existed only in holdout; added as tuning combos so
   debugging never touches the holdout).

AGENT: SLIDE CONCEPT landed (sim-discovered): is_slide = one direction,
spread magnitudes, WITH dominance guard (a step action with stray
double-magnitude noise must not classify — the unguarded version broke
cn04 to 0 levels; caught by the invariant sweep, fixed, re-verified).
Slide planning delta = direction x arena (sim-walk stops at known
walls = the slide rule); blocked slides mark the ADJACENT cell.

NUMBERS: sim tuning 17/24 L1 (mean first-clear 22.2), holdout 9/18
(mean 100.7) — the composition gap is now a measured curve to close.
Weakest combos: slide x collect 0/3, slide x hazard 1/3 (tuning-
debuggable next). DEV invariants all green: lp85 15/34/38, cn04 247,
sp80 11, m0r0 52, r11l 163, hazards 10/10 @95.28%, 14 tests.
Bundle v16 (4536 lines) validated repo-free, committed to user machine,
ledgered as the next submission (generic-layer delta hypothesis).

---

## Session 17 addendum (2026-08-29) — v16 SUBMITTED; slide round 2 (bundle v17)

v16 submitted to Kaggle (Version 6, scoring at session time) — ledger row
standing. Work continued immediately (waiting for scores buys nothing).

Slide concept, second round — four fixes, every one sim-discovered and
invariant-checked:
1. DISCRIMINATOR v2: slide = one direction + >=2 magnitudes EACH with
   real support (frequency dominance misread a much-used corridor as "a
   step of 11"; cn04's noise still excluded — its stray magnitude has
   single-vote support).
2. FREE WALL PER SLIDE (Law 3): a slide stops only where the next cell
   is blocked — every endpoint donates one wall for nothing (slide games
   otherwise starve the wall map).
3. BLOCKED-ATTRIBUTION PLAYFIELD GUARD: a blocked move whose full-vector
   target lies outside the SEEN playfield certifies only the ADJACENT
   cell (a long unclassified slide at a border marked phantom letterbox
   walls forever while the real wall's color went unlearned — the st80
   corner livelock: 196 identical doomed presses).
4. SLIDE-SCOPED COLOR TESTIMONY: cells crossed by a classified slide are
   proven passable but give no walked-on color testimony (a slide over a
   goal condemned the GOAL COLOR as terrain and goal-seek went blind).
   The broad version (all multi-cell moves) broke cn04 twice — terrain
   starvation diverged its fragile rotation-detection sequence — caught
   both times by the invariant sweep and scoped down.

NUMBERS: sim tuning 19/24 L1 (35 levels), holdout 9/18 (18 levels; was
8/16 at v16). DEV green: cn04 247, sp80 11, r11l 163, m0r0 52,
lp85 15/34/38, hazards 10/10 @95.28%, 14 tests. Bundle v17 (4571 lines)
validated repo-free; on the user machine; ledgered for tomorrow's slot.
OPEN (named): slide x collect stop-on-goal routing (st80 0/3),
drift x collect (holdout 0/3, concept pair missing from tuning — add);
then lf52 conversion.

---

## Session 17b addendum (2026-08-29) — V6 scored 0.27: the click-tax found and removed (v18)

V6 (v16) scored **0.27** — BELOW V3's 0.30. Honest verdict: everything
added v12-v16 netted NEGATIVE on hidden games. Diagnosis: the permuter's
phantom-button churn taxed every click/physics game (r11l 46 -> 163 =
~12x per-game score loss under quadratic scoring); hidden sets are
click-heavy, and lp85's L2/L3 gains covered fewer games than the tax
cost. Tushar's standing bar: NO submission until a bundle is PROJECTED
to beat 0.30.

THE FIX (one concept, from our own laws): a button exists only when
VERIFIED BY USE — a second consistent click on the SAME key confirms its
operator (cert counts too). Real buttons sit still and get re-clicked;
a moving rope never yields a second consistent observation on one key.
r11l: 163 -> 46 (best ever). Two induced regressions found and fixed
the same hour, both epistemically interesting:
- lp85 L3 38 -> 1146 NOT from the gate (certs were all present by
  action 100) but from PIECE-CLASS STARVATION: the goal tiles' diagonal
  single moves were dropped by the adjacency filter and the classes
  (11,4)/(12,4) never existed for the solver. Fix: STRUCTURAL piece
  registration — a standard-piece-sized object of a MARKER color on a
  track slot IS a piece; class existence must not depend on catching one
  clean move. (An eager version registered junk classes and broke L2 —
  scoped to marker colors.)
- L3 now 36 actions (better than the old 38); L2 41 (was 34).

Bundle v18 (4609 lines) validated repo-free; on the user machine;
ledgered with an explicit >0.30 projection. All invariants green
(hazards 10/10, five DEV wins, 14 tests, sim tuning 19/24 / holdout
9/18). NEXT for "drastic": convert zeros (lf52 first) so the projection
compounds before the next slot is spent — the drastic lever remains
the 15 DEV-zero games, not deeper wins.

---

## Session 17c addendum (2026-08-29) — lf52 CONVERTED (game family #6: peg solitaire)

"Keep improving" executed. lf52 decoded EMPIRICALLY this time (the
engine is 5,872 obfuscated lines — frames rendered as images and probed
by hand instead): an L-board of tiles, five pegs; clicking a peg selects
it (turns green, legal jump landing ringed); clicking the hole two cells
away jumps over the neighbor and REMOVES it; win = one peg left. Peg
solitaire — a classic family the hidden set may well contain.

NEW ARCHETYPE agent/pegs.py + mapper _peg_step:
- lattice extraction (largest identical-tile class -> pitch/origin;
  occupied tiles split into fragments and drop out of the class, so the
  board is tile cells UNION peg cells synthesized from lattice
  arithmetic);
- in-head DFS to one peg (memoized, deepest-reduction fallback);
- VERIFIED BY USE before engagement (the V6 lesson, now structural):
  one two-click experiment must visibly remove a peg; <=6 clicks worst
  case on non-peg click games; refuted thereafter;
- ATOMIC JUMP PAIRS: an engaged exchange takes priority over probing —
  a stray ACTION7 (undo) between the two clicks cancels the selection
  (observed: the probe sequence broke the first experiment).

RESULTS: lf52 L1 in 50 actions (agent; scripted optimum 8 clicks —
probe overhead is the gap, efficiency pass later). Full DEV sweep: lf52
is the ONLY change vs baseline (18 levels total). Hazards 10/10
@95.28%, 14 tests. Bundle v19 (4801 lines) validated repo-free,
committed to user machine, ledgered as the COMPOUNDED next-slot
candidate (v18 projection + a new family).
lf52 L2+ open (bigger boards; solver holds, perception to verify).

---

## Session 18 addendum (2026-08-29) — THE SCALING AUDIT (Tushar's directive: solid work, no band-aids)

Directive: "If you solved a level, why can't you solve the bigger
playground? There is a problem in our basic learning and/or hypothesis
and/or perception/execution." Audited every same-game scaling failure —
the highest-signal cases, because the mechanics are already KNOWN, so
each failure isolates a foundational defect.

THE META-LESSON (remembered, with its reason): **ARCHETYPES MUST
QUANTIFY OVER SETS OF PARTICIPANTS, NEVER OVER FIRST-OBSERVED COUNTS.**
Level 1 teaches a concept in its minimal form; bigger playgrounds
multiply the PARTICIPANTS, never the rules. Every hypothesis that
hard-codes the level-1 cardinality breaks on scale.

Audit table (cause classification):
- lp85 L4 (16 buttons): HYPOTHESIS — the ">8 uncertified buttons = not
  a button game" cap was a band-aid; per-key verification-by-use is the
  principled protection. CAP REMOVED; r11l stays 46 (the gate holds
  without it), lp85 L1-3 intact. L4 still open (16-key confirmation at
  57x57 is slow — efficiency, not correctness).
- lf52 L2 (two boards + pipe): PERCEPTION — one global lattice fused
  both boards and invented phantom pegs. FIX: a board is a CONNECTED
  lattice component (union-find on tile adjacency; fit pitch per
  component; pegs restricted to the component's extent). Bonus: cleaner
  perception halved L1 to 25 actions (was 50). L2's pipe-TRANSFER
  mechanic is genuinely novel — queued for the experiment machinery,
  NOT patched.
- sp80 L2 (3 basins, several platforms): HYPOTHESIS NARROWNESS — the
  aligner hard-codes one deflector/two targets/one flow direction. Needs
  the set-quantified rewrite (flow simulation over N pieces); own
  session.
- m0r0 L2 (divided halves + hazard strips): EXECUTION/BUDGET under
  scale — to instrument next.
- SIMULATOR TODO: add a SCALE axis (every archetype tested at 2x
  participants/board) so "solved the level but not the bigger
  playground" becomes a measured metric, not a surprise.

All invariants green after both fixes: six DEV wins (lf52 25, r11l 46,
lp85 15/41/36, cn04 247, sp80 11, m0r0 52), hazards 10/10 @95.28%,
14 tests, sim unchanged (tuning 19/24, holdout 9/18). Bundle v20
(4838 lines) validated repo-free, on the user machine — supersedes v19
as the next-slot candidate.

---

## Session 19 addendum (2026-08-29) — scale axis live; five foundational concepts; cn04 open

SIMULATOR SCALE AXIS BUILT (the audit's promised metric): 20x20 boards
at 3px cells, collect5, six hazards — 6 scaled combos (3 tuning /
3 holdout), 60 games total. First run: plain navigation scales 3/3;
slide degrades; collect x hazards at scale was 0/3 — then the fixes.

FIVE REMEMBERED CONCEPTS this session (each with its reason):
1. MOTION ATOMICITY: a step-action is a transaction evaluated at its
   DESTINATION (bounding-box engines teleport the full step). Partial
   progress fabricated off-lattice states (phantom frontiers, planner
   livelock); requiring passable intermediates blocked cn04's
   decor-colored corridors. Only slides move partially.
2. GOAL TOLERANCE + ARRIVAL: "go to X" on a lattice = the reachable
   cell nearest X; ALREADY-within-tolerance = arrived (equidistant
   lattice cells otherwise alternate as "nearest" and the avatar
   bounces forever).
3. COLLECTIBLE CLASS: moving onto a small object that vanishes — no
   level-up, no death — marks its COLOR as a collect-them-all goal
   class (the "unique odd object" goal rule had hard-coded cardinality
   1). Computed from pre-action objects directly: the anti-glitter
   protection suppresses diff events exactly where collection happens,
   and _prev_objs is overwritten mid-observe (snapshot required).
4. LETHAL CLASS: a death blames the NEAREST small object one
   movement-step from the final recorded cell (the trajectory ends one
   step short — observe skips GAME_OVER frames); walls and background
   never candidates (they block, they don't kill); NEVER on timer
   deaths (time killed us). Nearest-only: a generous radius once
   condemned a goal.
5. REVISABLE IDENTITY + STICKY TIMERS: a timer bar adopted as avatar
   before its shrink history matures is revoked retroactively; timer
   confirmation uses monotone SEGMENTS (bars refill on death — global
   monotonicity made confirmed timers flicker out) and is permanent.
   Identity anchors on the LARGEST controlled mover (vote-count
   anchoring self-reinforced a rigid body's tiny attached parts).
Plus: dock nav gain = distance actually removed (overshoot-rewarding
gain dithered 92-up/94-down at parity-offset poses); bounded nav steps
per pose (convergence contract).

MEASURED EFFECT: sim tuning 26/33 L1 (48 levels), **HOLDOUT 12/27 ->
19/27 L1 (22 -> 33 levels)** — the generalization gap closing is now a
number. m0r0 52 -> 41. All DEV wins green EXCEPT cn04 (0, was 247):
its docking pipeline survived identity healing (plug adopted, rot90
detected, dock engaged — all now correct-by-construction rather than
by trajectory luck) but the endgame no longer completes on this
trajectory. Classified honestly as needing its own deep session (3rd
fragility incident; breadcrumbs: pose (0,(20,29)) chosen, nav engages,
oscillation source was plan-tolerance ties now fixed, remaining
suspect: pose reachability after rotation-translations / port-exact
arrival vs tolerance).
KAGGLE CANDIDATE UNCHANGED: v20 (cn04 green there) stays the bundle on
the user machine; today's foundations ship in the repo and enter the
bundle after cn04 recovers. Hazards 10/10 @94.86, 14 tests.

---

## The generalization diagnosis (2026-08-29, Tushar's question: training, learning, skills, or approach?)

NOT training (none by design). NOT execution (with the right hypothesis
the agent is human-level or better: lp85 15/41/36, sp80 11, lf52 25).
NOT the learning machinery (laws converge; holdout 44% -> 70% this week
once the right concepts existed to bind).

THE GAP IS HYPOTHESIS CONCEPTION. The system's true shape: the
developer (Claude) is the META-LEARNER — reads a frame, recognizes
"peg solitaire" in seconds from thousands of culturally-acquired game
schemas — and the agent is COMPILED KNOWLEDGE, ~8 schemas deep. The
meta-learner does not ship (Kaggle rerun is offline). When a hidden
game rhymes with nothing compiled, the agent wanders: that is the
2-level naked-scientist cold-start, and it is the entire leaderboard
lag in one number. (Also why "knowing the past exams" is worthless
here — hidden games share concepts, never answers — and why brute
force cannot score: the quadratic penalty prices a grind at ~0.)

STRATEGY, three tiers:
1. COMPILE (proven point-scorer): keep converting families — breadth.
2. ABSTRACT (schema > birth-game): permuter's arc from "the lp85
   trick" to "clicked operators over geometric tracks" is the model —
   widen each schema past the game that taught it.
3. INDUCE (the research bet / paper core): a schema-induction layer —
   enumerate candidate rules from raw object RELATIONS (vanishes-on-
   touch, toggles, mirrors, carries, counts) and test them by
   experiment, so the agent can coin schemas never hand-written. The
   simulator is its training ground: generate mechanics deliberately
   OUTSIDE the coded ontology and measure whether the agent ever finds
   them. Collectible/lethal classes (session 19) are the first two
   induced-relation prototypes — the pattern scales.

## Session 20 addendum (2026-08-29): the docking endgame rebuilt

cn04 was 0 after the session-19 foundations (247 in v20). The failure
was not one bug but a CLASS of bug: every layer of the dock re-derived
the world from perception each step, and perception is exactly wrong at
the moment of contact — which is the goal state of docking. Five
concepts, each with its reason, all now in `_dock_step`:

1. CONTACT IS THE GOAL, NOT A MERGER. Socket discovery runs only while
   no pose is committed; the socket's footprint is frozen into the pose
   (`sock_pos`, grown UNCAPPED so the frame the notches sit on is
   included). Re-deriving the socket every step made the dock abort —
   and `_switch_piece` even killed the pose and rebound identity —
   precisely when the piece got close enough for rigid-group growth to
   swallow the socket. Reason to remember: any archetype whose GOAL is
   proximity must freeze its target representation at commit, because
   proximity corrupts the perception that found the target.

2. PORTS MATE BY OVERLAP -> NAVIGATE ON THE BODY. Connectors land
   exactly ON the socket's port cells at the pose, so a full-piece
   template can never match at or near the goal (observed: orientation
   missed one 3x3 port block, killing the pose one step from the win;
   the co-movement veto also dropped the piece's own fused port as a
   "touching neighbor"). Orientation, top-left, and nav distance all
   use body-only cells (they land in the bg cavity and stay clean);
   pose tl is translated to body coordinates at commit (`tl` body,
   `tlf` full for refutation keys).

3. COLOR-AWARE FOOTPRINT SUBTRACTION. While entering the cavity the
   piece legitimately covers former socket cells (it renders on top).
   Position-only subtraction ate the body mid-approach (piece_lost,
   eternal re-commit of the same pose). A footprint cell showing a
   piece BODY color is the piece; only non-body colors are socket.

4. SUCCESS IS A RECOLOR, READ AT THE SOURCE. With ports overlapping,
   group-derived port counts cannot distinguish "connected" from
   "sitting on top". The frame is read directly at the expected port
   positions (`pw`): a new non-port non-bg color = connected; port
   color = not connected (wait then refute); bg = pose missed the
   ports. Generalizes: when an event's evidence is occluded by the
   event itself, read the raw substrate at the predicted location.

5. REFUTATION KEYS MUST MATCH THE GENERATOR. `_dock_refuted` entries
   are consumed by `best_poses` keyed on FULL-piece (k, tl); after the
   body-tl change every refute site writes (k, tlf). A refutation
   ledger that silently keys differently from its consumer refutes
   nothing (observed: same pose re-committed 2742 times).

Result: cn04 L1 = 170 actions (v20: 247; pre-fix HEAD: 0), all other
DEV wins and sim holdout unchanged. L2 remains open (new click-probe
loop at (23,11) — next session). Positions-only orientation fallback
added in docking.py (port recolor must not break rotation reading).

Bundle v21 = v20 + session-19 foundations + this endgame. Ledger row
written; NEXT SLOT.

### Session 20b: v22 (identity integrity) + the L2 frontier

v22 (NEXT SLOT) adds two GLOBAL identity fixes found via cn04 L2:
ROTATION PRESERVES THE SQUARES (a shape-op action kills the body's
shape-keyed track while 3x3 fragments keep theirs — the tracker then
migrates identity onto a port; rebinds shrinking the avatar >3x on a
shape-op step are re-anchored by size/color/position continuity) and
SWITCH-ONLY-AFTER-SUCCESS (_switch_piece rebinds identity only after a
pixel-verified dock success; transient no-socket frames scattered
identity across remote fragments). Side effect: cn04 L1 170 -> 105.

cn04 L2 remains OPEN. What is now known:
- L2 is MULTI-PIECE: at least two port-bearing pieces ((0,90-cell) and
  (15,90-cell)) plus 9-cell port fragments; identity thrashes between
  pieces on ordinary moves because in select-then-act games EVERY piece
  is "the largest controlled mover" at different times. The avatar
  concept itself is the wrong abstraction here — the dock needs its own
  persistent piece registry (which piece am I docking, where was it
  last), not the global avatar.
- NEGATIVE RESULT (tried, reverted): pinning the dock's seed to the
  committed (color,size)-nearest-centroid made L1 WORSE (105 -> 164):
  the identity-based seed's d>20 abort was silently acting as a
  transient-frame filter — on thrash frames the dock paused instead of
  acting on corrupted perception, and the pin removed that pause, so
  arrival/refute judgments fired on bad frames and refuted true poses.
  Lesson worth remembering: an abort path can be load-bearing evidence
  filtering; replacing it needs an equivalent frame-quality gate
  (e.g. act only when the seed's group matches the committed base).
- L2 also needs pose-space reasoning: after each rotate press the
  base renormalizes, so the same physical pose reappears under a new
  (k, tl) key and the refutation ledger does not transfer. A
  rotation-invariant pose key (physical socket cavity id + piece
  orientation) would let refutations accumulate across commits.

### Session 20c: v23 — the docking archetype hardened for the unseen

Five generalizations shipped (see ledger v23 row), each born from a cn04
L2 failure but stated game-independently: footprint-keyed refutations,
pinned-seed + explicit clean-frame gate, click-a-cell-not-the-centroid,
full-template orientation with body fallback (symmetric bodies
degenerate), immediate clean-arrival refutation.

cn04 L2 STILL OPEN — the precise remaining problem is now measurable:
WRONG poses read as pixel-success (expected-port positions land on
frame-colored cells -> "recolored"), so they are neither refuted nor
completed and the pose search cannot converge. The obvious filter
(judge only positions that were socket ports at commit) BROKE L1:
at L1's true pose the coincident subset apparently does NOT read as
recolored even though the full 18-cell read does — meaning we do NOT
yet understand the engine's recolor geometry. NEXT SESSION, FIRST:
dump the actual frame at L1's docked pose and map exactly which cells
recolor to what; only then redesign the success predicate. (Lesson
recorded: a success predicate must be validated against a KNOWN
success before it is used to refute unknowns.)

### Session 20d: cn04 decoded at the source — the L2 playbook

Ground truth established two ways: frame capture at L1's win (the level
wipes on the connect frame — NO recolored intermediate renders in a
single-dock level; the engine's level-up IS the signal there), then the
DEV engine source (fair game, as with the harness; hidden games share
concepts, never code). What cn04 actually is:

- PORTS HAVE POLARITY: two port kinds (colors 8 and 13 internally),
  RENDERED IDENTICALLY. Display recolors ANY coincident pair to the
  connect color; the WIN counts only same-kind pairs. A visually
  perfect dock can be polarity-wrong — the agent must be ready to undo
  and reassign pairings when "everything looks connected" yet the
  level does not advance.
- CONNECT COLOR: coincident port pixels render color 3 — a color that
  exists nowhere else in the level. Hence v24's predicate: success =
  out-of-palette color at expected port positions, all of them.
- SELECTION LIVERY IS REAL: the selected sprite's body renders color 0;
  unselected sprites keep their native colors. "The piece is color 0"
  was a lie of perception — 0 means SELECTED. The parked livery-seeding
  idea deserves revival as a learned rule (clicked sprite adopts the
  selection color).
- PER-LEVEL ACTION BUDGET: MaxSteps (~150) then lose+reset. The 4000-
  action L2 searches were silently resetting every 150 actions. Pose
  search must fit the budget; footprint-keyed refutations SURVIVE the
  reset (agent memory, position-based, layout resets identically) —
  that is the compounding resource across attempts.
- L2 IS MULTI-SPRITE DOCKING: several movable port-bearing sprites
  (selection switches control); win = every port pixel of every sprite
  paired same-kind. The "one piece, one socket" frame is too small —
  needed: a port-graph assignment layer (which sprite's ports pair
  with which), then per-pair docking with the existing machinery.

NEXT SESSION: build the assignment layer for multi-sprite docking
(enumerate port-group pairings by geometric compatibility, dock pair
by pair, use the connect-color signal per pair, undo/reassign on
polarity failure), with the 150-action budget as the planning constraint.

## Session 21 (2026-08-30): THE BREADTH RELEASE (v25)

The multiple-fold thesis: family coverage is the score's primary driver.
One session, four zero->win conversions (parallel source-decode by five
diagnosis agents, then serial gated-archetype implementation with the
battery after every change):

- tu93 0->3: EDGE-MAZE archetype. Concepts: passability can live on
  EDGE MIDPOINTS, not cells; moves can be a metered budget (never emit
  a move the model calls blocked); an archetype may need SELF-ANCHORED
  perception when the generic mover attribution is poisoned; action-map
  calibration + 4-way elimination.
- ft09 0->4 (@4/7/14/16 actions, near-optimal): CONSTRAINT-GLYPH
  archetype. Concepts: a glyph tile's pixels can encode equal/differ
  constraints on its neighbors; boards ship with decorative LEGEND
  copies — cluster lattices and verify each by use (one free click).
- sb26 0->1 (@9 actions, optimal): REFERENCE-ASSEMBLY archetype.
  Concepts: protocol games (select->place->submit) cannot be stumbled
  into; roles = reference strip + tray + slots; trays can live BEYOND
  the separator line playfield detection reads as a border — scan the
  full frame; protocol archetypes must preempt probing (the pegs
  lesson, again).
- s5i5 0->1: REMOTE-ACTUATOR archetype — the first EXPERIMENTAL
  PLANNER: no avatar, learn click->delta operators by experiment,
  hill-climb marker-to-target distance, recency-weighted bandit
  (a saturated spot's glorious past must not earn it budget).

Generic-perception lessons (each recorded with its failure):
- WAKE PHANTOMS BELONG TO A CROWD (cover/uncover pairs outsize the
  body and invert votes; crowd-color movers neither vote nor anchor).
- SHAPE-MORPHING MOVERS (facing-flip bodies read as appear/vanish and
  the 1px indicator inherits the vote; pair same-color near pairs,
  distance-capped after a cross-map collect-pair poisoned a table).
- RECURRING AMBIENT EVENT CLASSES (fog redraw fragments kept the
  blocked-verdict from ever firing; walls never learned).
- A GATED GOAL YIELDS TO FREE INFORMATION, WITH DECAY (state-match
  pads: two refusals gate the goal color, a new touch or 40 quiet
  steps clears it — refusal may be timing, not condition).

PARKED (findings recorded, archetypes gated + harmless): re86 overlay
(~85%: control flapping between engine selection and belief; candidate
offsets now provably correct each step — needs selection-belief rebuilt
around "the engine's selected shape is the last mover"), ls20 (fog +
budget + delivery stack; ambient fix landed, planner loop remains).

DEV portfolio: 19 levels/6 families (v20) -> 29 levels/10 families.
Holdout dip 19->18 investigated and cleared: shg0 fails identically on
the v24 snapshot (pre-existing noise, not today's changes).
Zero queue remaining: dc22 (layout-switch clicking, MED), ka59
(impulse-push multibody, MED-HIGH), sk48 (chain-avatar, MED-HIGH),
bp35 (scrolling dig-down, HIGH), ls20, re86.

### Session 21b: v25 SUBMITTED; v26 staged (layout-switch supervisor)

v25 went to Kaggle (2026-08-30). v26 is on the machine for tomorrow's
slot: the LAYOUT-SWITCH SUPERVISOR from the dc22 campaign — concepts:
THE WORLD CAN BE REWRITTEN BY A BUTTON (a confirmed layout rewrite
invalidates every learned wall: model.forget_terrain), buttons live
beyond the playfield crop (full-frame scan — the sb26 tray lesson
recurs), button identity is POSITION (their glyphs display state and
change on click), and a good new layout needs SETTLE TIME before the
supervisor may click it away.

dc22 STILL OPEN — precise remaining gap: SUPPORT-CONDITIONED MOVEMENT.
Passability there is "destination overlaps a platform sprite", not
wall-colors; after forget_terrain the walker relearns only locally and
bounces. Next session: learn "support colors" (colors under cells the
avatar successfully occupies), predict passability from FRAME CONTENT,
feed plan_to a frame-derived passable mask. With that, the supervisor's
button-cycling + replan loop should clear dc22 L1-L3.

re86 parked again (3rd pass): tracking now survives dot-parse failures,
but late-game drive direction oscillates (candidate offset flips sign
frame to frame — suspect stale dot-memory entries interacting with the
covered-dot logic). Needs a fresh-eyes session, not a fourth patch.

Portfolio unchanged from v25 (28-29 levels / 10 families); supervisor
battery-verified as zero-regression.

## Session 21c (2026-08-30): V7 = 0.21 — THE FALSIFICATION, AND THE LAW

The hidden set ranked our whole line: v11 = 0.30 > v16 = 0.27 >
v25 = 0.21. Every DEV gain since v11 NET SUBTRACTED hidden points.
This falsifies the naive breadth thesis: family coverage only pays if
the added machinery is FREE on games that do not match it, and ours
was not. Three mechanisms, in likely order of damage:

1. MISFIRE TAX (the r11l click-tax, generalized): an archetype whose
   gate trips on an unknown game spends tens-to-hundreds of actions
   failing; RHAE is quadratic, so one hijacked game loses nearly its
   whole score. The overlay/actuator/refassembly gates were tuned to
   not misfire on OUR 16 DEV games — the battery is structurally blind
   to off-distribution misfires.
2. CRASH STORMS: any exception inside an archetype made the harness
   emit a RANDOM action — on a frame shape we never saw locally, one
   brittle numpy index could turn a whole game into noise.
3. GENERIC DRIFT: the v17-24 generic changes (slide machinery,
   foundations, identity work) were never individually scored; some
   may redirect play on hidden games (e.g., a mis-blamed lethal color
   can exclude the true goal).

DESIGN LAW (permanent): FAIL-CLOSED. v27 implements it — a central
governor meters every archetype per level (allowance = 1.5-2x its DEV
need); overspend without a level-up = silenced for the level; ANY
exception = silenced. Plus: a lethal condemnation that blanks the goal
set is ignored. All 28 DEV levels verified identical under the
governor.

MEASUREMENT PLAN for the next slots (one bit per day, spend wisely):
- Next slot: v27. It cannot do worse than v25 (same capabilities,
  strictly bounded downside). Its score SPLITS the tax: if v27 >=
  ~0.27, the loss was archetype misfire/crashes (now capped) and the
  conversions can start paying; if v27 stays ~0.21-0.24, the tax lives
  in the GENERIC drift of v12+, and the following slot should be a
  reconstructed-conservative bundle (v11-style core + governor +
  only the provably-passive additions).
- Kaggle retains every submitted notebook version: Version 3 (~v11,
  0.30) and Version 6 (v16, 0.27) can be recovered from the kernel's
  version history for exact diffing if reconstruction is needed.
- Final-score safety: Kaggle auto-selects the best 2 submissions
  (currently 0.30 + 0.27), so experiments cannot hurt the final
  standing — the slot's only cost is a day.

## Session 22 (2026-09-03): V8 = 0.21 — the drift is GENERIC; v28 staged

RHAE's own math localizes the loss: actions on unfinished levels never
hurt banked levels, so archetype misfires on losing games cost ZERO.
0.27 -> 0.21 = hidden levels v16 won that the v17+ ALWAYS-ON layer now
loses. v28 (staged) reverts the prime suspect: the v21 arrival fake
(plan_to returning [] "within tolerance" — up to a full step short of
goals that must be TOUCHED), replaced by deterministic nearest-cell
tie-breaking; lethal condemnation now needs two witnesses; and the
HYPOTHESIS LEDGER (agent.hlog) records every adopted belief with the
alternatives that were also consistent (Tushar's requirement: when
multiple genuine answers existed, log that we knew, and what we chose).

THE FOUR QUESTIONS (Tushar, 2026-09-03) — answers pinned:

1. WHAT HARDENING BACKFIRES: candidates ranked — (i) arrival fake
   (v28 reverts), (ii) motion atomicity's destination-only walks,
   (iii) slide-v2 attribution (free-wall-per-slide can invent walls on
   misclassified actions), (iv) largest-mover identity + revision
   (avatar flapping on hidden sprites), (v) one-shot lethal/collectible
   condemnations (v28 softens lethal). Each was correct ON ITS BIRTH
   GAME; the backfire mode is always the same: a rule with a
   game-specific precondition firing where the precondition is unseen.

2. EXPLORATION & HYPOTHESIS CREATION: exploration is adequate
   (probe-first, touch prior, frontier); hypothesis CREATION is the
   weak organ — we FORM few hypotheses and COMMIT early ("laws of
   cheap learning": single-vote beliefs persisted until contradicted;
   on a hidden game the contradiction may never arrive). The
   micromouse era felt better because run 1 was allowed to be 100%
   exploration; the single-pass pivot made us stingy with evidence.
   Direction: hypothesis PORTFOLIOS (carry 2-3 live candidates, act on
   the one with best expected value, log the others — hlog is step 1).

3. UNIVERSALITY (pixel vs object): perception hard-commits to
   same-color connected components at one scale. Facing dots, 1px
   selection markers, multi-color rigid pieces, port polarities — each
   needed a hand patch (morph movers, port re-attach, livery). The
   universal fix is MULTI-SCALE ENTITIES: track pixel / component /
   rigid-group / class simultaneously and let control-correlation
   evidence pick the level that predicts best. Paper-worthy refactor;
   schedule as its own session.

4. WHY NO PATTERN FOUND + ANSWER-KEY STUDY: begun via hlog (belief vs
   alternatives). Next: an offline study tool that replays DEV losses
   against engine-source ground truth and emits, per game, the
   DIFFERENCE between what the agent believed at each decision and
   what the source says was true — hypothesis classes to induce come
   from those differences. (We already run this manually per game;
   formalizing it is the tier-3 induction training set.)

## Session 22b (2026-09-03): THE v11 STUDY — measuring instead of guessing

Tushar's principle, adopted: a low score does not mean a failed
concept; we learn what works, we do not game the system. So no blind
reverts — the v11 notebook (Kaggle Version 3, 0.30) is now in the repo
(reference/my_agent_v11_kaggle_v3.py) with two study tools:
tools/run_reference.py (run any reference bundle on DEV with our
runner) and tools/lockstep.py (run two agents in lockstep on one game
and print the first divergent action with both agents' beliefs).

FINDINGS:
- On DEV, current dominates v11 everywhere (28 vs 13 levels) except
  mz02 (hazard CURRENT) L1: v11 68 vs current 130.
- Lockstep on mz02: identical beliefs at step 9; the PLANNER differed.
  v11 declined an unreachable target and kept probing; session-19's
  goal tolerance planned to the nearest cell on two drift-contaminated
  vectors. Concept: UNREACHABLE IS INFORMATION — approximation turns
  an epistemic signal into premature action. Tolerance is now an
  explicit opt-in for docking-class games only. mz02 68, cn04 99.
- Slide discriminator conflated slides with CURRENTS (both vary
  magnitude): drift is subtracted before judging magnitudes.
- Hypothesis ledger paid off on day one: water adopted as a
  collectible (COLLECTIBLES ARE SPARSE rule), goal choice flapping
  among equally defensible colors (COMMIT UNTIL REFUTED rule).
- A permuter-gate loosening was tried and refuted by data (r11l 46 ->
  176): recorded as a refuted hypothesis. The click-tax lesson stands.

WHY THIS MATTERS FOR FLUID INTELLIGENCE (not just the score): every
finding here is about the agent's EPISTEMICS — when to act on a
belief, when a failure to plan means "learn more", when a class is too
common to be a goal, when to stop re-deciding. Those are the reasoning
habits a fluid agent needs on ANY game. The score is downstream.

NEXT: (a) v29's verdict; (b) extend the lockstep study to the sim
holdout games where v11 and current differ; (c) the answer-key study
tool (belief vs engine truth per decision) on DEV losses; (d) the
multi-scale entity refactor (pixel / component / rigid group / class).

## Session 22c (2026-09-03): THE ATOM'S INSTRUMENTS

Design doc (project: concept-simulator-design.md) now carries the atom
definition (entity scales, relation grammar v0, evidence rules) and the
measurement suite M1-M9. First instrument shipped: tools/beliefaudit.py
— beliefs vs generator truth per sim game. Baseline readings:
  control prec 0.94 tuning / 0.82 holdout (diagonal directions on axis
  actions in slide x rotated-control pairings = a belief-space
  composition gap); lethal_ok 0.79/0.59 (the two-witness rule's recall
  cost, now quantified); slide_ok 0.67/0.63; flip-flops 0.2/0.7;
  stagnation on losing games ~390/400 steps (the hypothesis generator
  runs dry — the strongest argument for the grammar).
NEXT: (1) fix the diagonal-direction belief flaw (M3) — a real fluid-
layer bug found by the instrument, not by a game; (2) P5 out-of-grammar
mechanics authored blind, to make M1 (coining rate) measurable;
(3) the multi-scale entity refactor.

# RECALIBRATION (2026-09-04) — BINDING RULES

Agreed after three identical hidden 0.21s and a no-holds-barred review.
North star unchanged: a winning agent — reached through a real core, not
through compiled lookup tables. Two branches, one kill list.

## Branch A — THE ATOM (research; ~80% of effort)
Build the induction core: multi-scale entities (pixel / component /
rigid group — only what these games need), a compositional relation
grammar, evidence rules, hypothesis PORTFOLIOS, and ACTIVE EXPERIMENT
DESIGN (the agent chooses the action that discriminates live
hypotheses cheapest — disambiguation under uncertainty is the fluid act).
Rules:
  A1. NO reading engine source of any game. The agent must earn its
      hypotheses; I may not pre-solve.
  A2. NO new archetypes. No DEV-zero conversion campaign.
  A3. The simulator (with BLIND out-of-grammar mechanics) is the
      training ground; M1-M9 (concept-simulator-design.md) are the
      scoreboard; the real games are the external audit only.
  A4. Every belief adoption logs its alternatives (hlog). Every
      version change is lockstep-attributable (M8).
  A5. Deliverables: (i) the loop stated formally with a convergence
      contract; (ii) the generator + blind coining test + belief
      metrics; (iii) negative results as first-class findings.
  A6. Success in six weeks: coining rate (M1) > 0 on blind mechanics;
      naked-scientist cold-start climbing without any compiled addition.

## Branch B — THE CONTENDER (competition; <=20% of effort)
Base = v29 (reference/contender_v29_base.py): v11's proven organs +
strictly additive governed families + the patience planner.
Rules:
  B1. A submission only when Branch A yields a learning that is
      provably bounded in cost (governor) — never a guess.
  B2. Every submission carries a PRE-REGISTERED PREDICTION in the
      ledger (score range + the mechanism it tests); the ledger records
      hit/miss. My calibration is a measured number.
  B3. Kaggle auto-selects the best two; B cannot lower the standing.

## Kill list (effective immediately)
  - the DEV-zero conversion loop
  - engine-source reading
  - celebratory projections without pre-registration
  - patching perception per game (the representation gets fixed, once)

## Thesis clarification (Tushar, 2026-09-04)
In scope: inferring meaning from illegible/ambiguous input, holding
multiple genuine meanings, acting to disambiguate. Out of scope:
perceptual invariance (recognizing a cat front/back). Perception need
only be good enough to feed the reasoning; the atom's job is
disambiguation under uncertainty.

# SESSION 23 (2026-09-04) — THE ATOM v0: CONDITIONED LAWS. M1 > 0.

Branch A only. Contender untouched (v29 still the candidate; every DEV
level action-for-action identical after every change below — M8).

1. Instruments corrected/extended
   - beliefaudit: control precision was 1.00 all along (the 0.82 was a
     double drift subtraction in the instrument); lethal recall is now
     DEATHS-CONDITIONED (a game with <2 deaths cannot be a lethal miss).
   - is_slide DISPERSION rule verified on DEV: 28/28 levels identical.
   - tools/blindrun.py: the coining test in two phases (record beliefs
     -> reveal sealed answers beside the record). tools/lockstep.py now
     takes any reference bundle (default: contender_v29_base).
2. BLIND SET 1 (sx01-06, authored by a separate session) — baseline
   M1 = 0/6. Score-level: L1 cleared on 3/6 by generic play (push by
   collectible-adoption luck; blinking gates by retry; click-swap by
   random clicking at 308 actions). Stagnation 370-400/400 on every
   game: the vocabulary had no slot for any of the six relations.
   SET 1 IS NOW UNSEALED = a tuning set (sx05 drove the mechanism below;
   sx01/02/03/04/06 remain open tuning targets).
3. THE ATOM v0 — CONDITIONED LAWS (world_model.Law):
   a contested belief ("ACTION2 moves down", "color 5 is floor") is
   neither discarded nor averaged; the contradiction is explained by the
   simplest CONTEXT STUMP that separates successes from refusals
   (features: action-count parity, previous move axis, previous action).
   Evidence bar: perfect separation, >=3 witnesses a side, refusals from
   >=2 distinct cells, no known wall on the step path, never under a
   current. Refuted by the first counter-example (logged). The planner
   searches the PRODUCT STATE (cell x the context features any law
   references) — with no laws it is byte-identical to before; a refusal
   the law predicts is a legitimate plan step (a bump that flips parity).
   Laws carry across levels; context resets on RESET/level.
   Support fixes the mechanism forced into the open, all generic:
   - terrain testimony comes from the color a cell showed BEFORE the
     avatar stood on it (floor had been condemned as wall by refusals);
   - an interactable counts as touched when REACHED, not when planned
     (three abandoned plans = give up; any new law re-opens everything);
   - COMMIT-UNTIL-REFUTED now has a refutation: a goal the fully
     explored map cannot reach is refuted -> alternatives (hlog
     goal_refuted).
   RESULT: sx05 (alternate-axis rule) 0 levels/400 -> 2 levels in 56
   actions; laws adopted at steps 14/62/66 of L1, reused in L2. M1 = 1/6
   on set 1 (post-hoc, set 1 is tuning now). mz02 false law caught by
   lockstep (M8 working as designed) -> evidence bar raised.
4. BLIND SET 2 (sx07-12) authored blind by a second agent, key sealed in
   BLIND_ANSWERS_2.json (NOT read). Phase-1 record (results/blindrun2):
   L1 cleared 1/6 (sx09, collectible adoption); zero laws coined; sx07
   is an all-slide game (the known slide weakness); sx10 learned NOTHING
   in 400 actions (no motion at all); sx11 motions inconsistent. The
   reveal is deferred until the next mechanism lands, so set 2 gives a
   clean before/after M1 reading.
NEXT (Branch A, in order): (1) conditioned OUTCOMES, not only refusals —
Law generalized to "in context C the action's vector is V" (covers
wrap-at-edge, sx02; and is the honest form of the slide rule);
(2) causal click laws: click(color Y) -> avatar delta (sx06 click-swap
is the tuning case; hidden games are click-heavy); (3) contact-conditioned
co-motion (push, sx01); then reveal set 2 and read M1.

## Session 23b (2026-09-04): CAUSAL CLICK LAWS; M1 = 2/6; DEV 28 -> 29

Second coined relation form, from the grammar's causal row: CLICK(X) ->
avatar lands on X (world_model.record_click). Two witnesses adopt, one
click on X that leaves the avatar put refutes; the planner gains a
teleport edge to every scarce X-instance (bounded: <=6 instances). The
BODY MUST BE DISTINCTIVE (<=2 same-colored objects) for a click to
testify — among r11l's rope pieces identity hops onto whatever was
clicked and every hop read as a teleport (false law at step 16, refuted
at 38, trajectory bent 46 -> 202); with the rule r11l is 46 again. Click
trials and laws carry across levels. sx06 (click-swap): 308 actions /
1 level -> 61 actions / 2 levels.

Exploration and identity epistemics that the blind set forced open —
each generic, each battery-verified:
- A FRONTIER IS WHERE PASSABILITY IS UNPREDICTED: once a floor color has
  three free witnesses and <=1/4 wall share, its cells are not worth
  visiting (tier 1); tier 2 is the exhaustive walk, needed where walls
  live between identical cells (tu93). Between them: up to four
  class-scored clicks per level. A "probe every scarce class" stage was
  tried and REFUTED (it preempted the class-scored click policy; r11l's
  click-tax returned 46 -> 437). Goal refutation by "map exhausted" uses
  the exhaustive tier.
- ARRIVAL WITHOUT EFFECT REFUTES A GOAL (standing on the sought object
  twice, nothing happened): the teleport anchor had been adopted as
  "the unique small object" and ping-ponged the avatar forever.
- IDENTITY IS REVISABLE BY NON-RESPONSE: a sticky avatar track that
  ignores four consecutive commands while another track answers loses
  the body; a track handed identity by direct evidence (a teleport
  landing under a standing law) holds it (tracker.bind_to / forced).
  A blanket "appearance beats position under clicks" matcher was tried
  and REFUTED (r11l 46 -> 436): appearance is identity evidence only
  when distinctive, and even then r11l's trajectory bends — left at the
  old position-first matching.
- Planner: is_slide hoisted out of the BFS inner loop (re86 450 actions
  109 s -> 19 s; behavior identical).

THE SIMULATOR CAUGHT THREE REGRESSIONS THE DEV BATTERY COULD NOT (M4
doing its job — tuning set only; holdout never inspected):
- terrain testimony from the pre-move color condemned COLLECTIBLES as
  terrain (st21 collect3: 29 -> 55, L2 lost): testimony now settles
  only when the vacated cell REVERTS to that color (a collected item
  leaves background);
- the map-exhausted goal refutation fired at step 6 of a slide/collect
  game against a class verified by use (st82 0 levels): never refute a
  collectible class by unreachability; require settled probes and >=20
  steps;
- the is_slide DISPERSION rule (from the belief audit) was a wash on
  the games (st72 0 -> 2, st32 2 -> 0, holdout L1 18 -> 17): reverted
  to the v29 rule; recorded as a negative result — a belief metric can
  improve while the games do not, when the planner cannot yet use the
  belief (M9's disagreement rule, first instance).

DEV battery (800 actions, seed 0) vs the pre-atom snapshot: ls20 0 -> 1
(27 actions — the game two archetype campaigns could not open, cleared
by exploration epistemics alone), cn04 99 -> 168 (the docking pipeline's
trajectory fragility, 4th incident: twelve strict-tier steps between
probe resolution and rotation detection change the walk; accepted in
Branch A, noted for the Contender), m0r0 46 -> 50, all else
action-identical (r11l 46, tu93 3, lp85 15/41/39, ft09 4). 29 levels.
The strict tier engages only once probes are settled and no shape-op
action exists.
SIMULATOR (M4, never inspected holdout): tuning 26/33 games, 46 -> 49
levels; HOLDOUT 18 -> 19 games, 33 -> 34 levels — the composition gap
moved the right way with zero holdout debugging.
Blind set 2 (still SEALED): L1 cleared 3/6 by generic play (sx09 2
levels, sx10 1) at the battery cap, laws coined 0 — the reveal waits
for the next mechanism.
Bundle v30 (7170 lines) validated bundle == modular on sx05/sx06/mz02/
r11l/ls20/st82 and repo-free exec; ledgered as the candidate AFTER v29.

## Session 23c (2026-09-05): THE OPEN HYPOTHESIS — "I don't know" as a first-class belief

Tushar's directive after V10 = 0.24: unless a theory is confirmed more
than once on a unique case, the program keeps an open mind that a new
mechanic may be present; same game / new level -> carried learnings
first, and only when they fail, live hypotheses. Built, all generic:

1. THE ANOMALY LEDGER (mapper._detect_anomalies). An observation that
   fits no relation the program can name is neither rounded to the
   nearest known concept nor dismissed as noise; it is recorded with its
   context and RE-TESTED (_retest_anomaly: return to where it happened
   and repeat the action, or step into the object again from behind;
   two re-tests per anomaly, twelve actions per level; a re-test where
   nothing moved is inconclusive, not a refutation). Kinds: UNEXPLAINED
   MOVE (a confirmed step action produced a different DIRECTION —
   magnitude-only differences belong to the slide/current machinery,
   a lesson from st82/st71), CO-MOTION (a non-avatar object moved with
   the avatar's vector from the cell in front of it), and contested
   refusals/outcomes with no separating context (law_unknown). A
   spontaneous repeat is a confirmation; two witnesses confirm.
2. CONDITIONED OUTCOMES (world_model._outcome_search): an action with
   two well-supported vectors is explained by a stump over the context
   features plus a positional one, AT_EDGE (the cell beyond has never
   been seen). Coined on sx02: "ACTION1 at the edge -> (15,0), else
   (-1,0)" — the wrap-around, from the grammar, nothing hand-written.
   Planner uses the outcome per state.
3. THE EDGE IS A HYPOTHESIS: once per direction per level, standing at
   the boundary, press the outward action (the planner never plans into
   unseen cells, so "the boundary is a wall" had never been tested; sx02
   found its wrap at step 189 by accident, now at step 14 by design).
4. THE PLAYFIELD IS A HYPOTHESIS (mapper._refute_playfield): the
   band-split that reads a full-length wall line as a HUD divider is a
   prior; when the body vanishes from the crop and stands, same key, in
   the excluded region, the crop widens to the frame (sx02's goal half
   had been cut off).
5. RELATION COINED: "pushable(color)" from a confirmed co-motion (sx01,
   step 8). The win condition (block on target) is not yet inferred.
6. Identity: a track NOT matched this frame no longer drags the avatar
   back to its stale last position (sx02's wrap law was being refuted
   by its own ghost). This one rule flips tu93 (3 -> 1 levels: the
   edge-maze archetype's L2/L3 trajectory depended on the stale drag)
   and ls20 (0 -> 1) in opposite directions — kept, because it is the
   correct rule; tu93's dependence is archetype fragility, logged.

BLIND SET 1 (tuning now): M1 = 4/6 coined (sx05 conditioned refusal,
sx06 click teleport, sx02 wrap outcome law, sx01 pushable), sx03
(mirrored twin) recognized as an unknown mechanic, sx04 cleared by
retry. sx02: 684 actions / 1 level -> 79 + 250 / 2 levels.
BLIND SET 2 (still SEALED, phase-1 record only): L1 cleared 4/6 (was
3/6): sx07 303 -> 80, sx08 0 -> 1, sx12 0 -> 2 levels, sx09 faster.
The reveal is still deferred.

AUDITS for this build (M9, honest): DEV 29 -> 27 levels — tu93 3 -> 1
(the identity rule), cn04 168 -> 122, r11l 46 -> 48, ls20 kept; sim
tuning 49 -> 46 levels, HOLDOUT 34 -> 32 (never inspected; the tuning
losses are slide games — st82, st71 — whose wins had depended on the
stale identity drag; the slide machinery is a coin flip under every
identity change and needs its own treatment: "moves until blocked" as
a conditioned outcome, next). Two more refuted design hypotheses,
kept: unique-color bodies matched at any distance in the tracker
(velocity EMA corrupted by jumps: sx02 and ls20 lost), and the
dispersion slide rule re-tried under correct identity (still a wash:
st72 up, st32 down). NOT a Contender candidate: v30 stays the bundle
after v29; this build is Branch A's research state (repo s23c).

## Session 23d (2026-09-05): THE FAILURE ATLAS — ground truth vs beliefs on all 24 real games

Tushar's directive: for every game, what winning looks like and where the
loop failed — from the engine keys, not from generic reasoning. Four
agents extracted per-game ground truth (study/truth/*.md, every claim
cited, every L1 winning sequence run and verified); the agent ran on all
24 with belief telemetry (tools/failure_atlas.py, study/runs/*.json).
The real-game HOLDOUT (8 games) was unfrozen for the study.
RESULT (study/FAILURE_ATLAS.md, project doc failure-atlas.md):
- HOLDOUT 0 of 57 levels on eight never-seen games whose human L1
  costs are 7-78 actions: the honest hidden-set proxy.
- Nine failure classes, ranked by how many games they cost: no goal-
  inference organ (18/24 wins are RELATIONS: cover, inside, aligned,
  equal-to-reference, sequence, mapping, count); unreachable misread as
  not-the-goal (withdrawn: candidates are now DEFERRED, not refuted);
  "which body do I control" absent (selection livery: cn04 ka59 re86
  r11l ar25 sp80 m0r0); facing turns read as rotations (sc25, wa30 —
  docking hijack); budgets read as deaths (all 24 games meter actions;
  sc25's door condemned lethal); one click law vs ten click semantics
  (select/toggle/pump/pull/rotate/delete/move-to-point) and a click
  policy with no causal model (m0r0 749, vc33 785, tn36 788 clicks);
  deferred-effect programs (sb26 tn36 cd82 sp80 sc25); perception priors
  that remove the game (playfield crop: dc22 panel, sx02 board half;
  largest-mover identity: bp35, vc33); compiled organs that stop at the
  first added relation (every archetype, every game).
- The priors a human uses on frame 1 and we lack: same colour ->
  belongs together; picture beside blank -> make equal; strip repeating
  the play glyphs -> reference; frame fits block -> put it in; table of
  pairs -> mapping; dotted line A->B -> take A to B; shrinking bar ->
  budget.
NEXT: goal induction from free information (hypothesis classes cover /
inside / aligned / equal / sequence / mapping / count from same-colour,
same-shape, same-size correspondences), selection-with-livery, facing
is not rotation, a universal budget reader.

## Session 23e (2026-09-05): THE GOAL ORGAN — goal induction from free information

Tushar: "so we thought the game was like pac-man or tetris whereas it is
much broader... is it a new goal-inference organ or something I already
thought about and it was not implemented as cleverly or broadly as it
should?" Answer: the latter, made explicit. The atom had "goal =
unique small object, reach it"; the atlas showed 18/24 wins are
RELATIONS read off the first frame. agent/goals.py is that reader,
integrated as the first step of act() (the core proposes, the level
ending disposes; archetypes and the world model keep their turn after).

THE ORGAN (agent/goals.py, ~750 lines; nothing in it knows a game):
- ENTITIES: clusters of same-colour components (gap <= 2); tiny
  components (<= 6 cells) grouped at gap <= 12 with >= 3 parts are
  CONSTELLATIONS (corner marks, dot patterns, outlines drawn as dots) —
  markers, never movers, their cells removed from the entity pool.
- READINGS -> ranked hypotheses (align | cover): (0) mover -> own-colour
  constellation; a constellation whose cavity fits the mover is a
  SOCKET (s5i5's dot and hollow diamond); (a) same colour, different
  KIND (congruence up to rotation/mirror, or two solid bars of one
  thickness, makes peers — a slider's two brackets, a panel's two rings —
  never mover and marker; a marker 10x the mover is a floor); a marker
  whose cavity fits the mover is a socket; (b) size fit into a frame's
  cavity across colours; (c) shape congruence with a constellation;
  (d) a TWIN IN THE HUD names the mover, an enclosing marker of a unique
  colour is where it goes (su15). Terrain colours (walked/bumped, area
  >= 24) are never movers. Cap 4 per colour, 12 per level.
- EFFECT LEARNING (GoalEffectMap, level-wide, one record per press =
  the motion of the colour's cluster that moved most; recent-3): keys
  do the same thing everywhere; a click ON A THING is a button (absolute
  key); a click NEAR THE MOVER is relative ('rclick', offset) so what
  was learned at one place transfers (su15's pull, 9/6/3 probes,
  longest first). HUD bars are outside the snapshot (the step bar moves
  at every action).
- DESCENT (GoalActuator): greedy on a distance — controllable axes only
  once every key has had its turn with THIS mover (vc33's one-axis
  relation; the pooled-colour version marked s5i5's dot 'arrived' when
  the OTHER dot slid); cover-distance is cell-wise when the marks lie
  within the mover's extent (re86's dots on the cross) and centre-wise
  when they frame a spot (lp85's corner marks). Exploration order is
  the order a person pokes: keys; the thing itself (once); beyond it
  toward the marker; the marker; the centre of every other small thing
  that has NOT moved under our own actions (buttons do not move); then
  the ends of elongated things. Exploration capped at 20 per hypothesis.
- MODES / SELECTION: a key that moved nothing large while something
  small changed (a livery, a centre pixel) is a mode candidate; the
  press already made IS the mode press, so the keys that moved things
  are re-checked at once: reproduced -> the press was nothing (ka59's
  blocked arrow costs one action); contradicted -> the world changed
  hands, the record is rebuilt (re86's ACTION5; r11l's handle click;
  ka59's block click). Old records are re-checked, never thrown away.
- ARRIVAL: a mover that merges with its marker (s5i5: the dot inside
  the diamond is one component) is tracked as the cluster containing
  the predicted cell; satisfied only after it has MOVED under this
  actuator; the level ending confirms; refutations: mover lost, budget
  40, stalled (20 actions without progress after the keys were tried),
  colour never moved (zero cost once the level's pass has 6+ tests).
- UNBLOCK: a key that carried the mover and is inert now — press a key
  known to move something ELSE, then retry (vc33 L2: the partner piston
  ran out of length). HANDOFF: when a gain key is refused here and the
  world model's body is the mover, the planner's route is followed
  (the organ names the goal, the planner reaches it; ls20 L1 confirmed
  this way; ka59 finds the route to the pit edge and no further).
- TRANSFER: a reading the level ending confirmed (the rule and which
  side moves) goes first on the next level of the same game.
Also this session: the single-known-motion random fallback (su15 spent
795 actions on the undo key); walked-over is not 'not the goal' —
small-area colours are not terrain (st71: a slide carried the body
across the goal, which became floor).

MEASURED (tools/failure_atlas.py, all 24 real games, 800 actions):
  22 levels vs 17 (s23d); games with L1: 15/24 vs 12/24.
  HOLDOUT L1: 2/8 vs 0/8 — vc33 L1 in 17 (human 7), su15 L1 in 10
  (human 22). New: re86 L1 41 (human 26), tu93 3 levels (was 1).
  Faster: r11l 12 (was 48), s5i5 29 (was 41), cn04 103 (122).
  Slower (the organ's exploration on levels an archetype already won):
  ft09 +10/level, ls20 44 (27), lp85 L1 35 (15; L2 31 vs 41).
  Blind set 1: L1 5/6 (was 4/6). Simbench: tuning 26/33 games, 49
  levels (= v30); holdout 17/27, 34 levels (v30 19/27 — shi0 and shi1
  were already lost at s23d, verified on the s23d snapshot; not
  inspected, they are holdout).
  In isolation (goaltest, 200-300 actions): vc33 L1 17 + L2 47 (with
  the unblock), lp85 3 levels, s5i5 L1 24-29, re86 L1 41-49, r11l L1
  12-13, su15 L1 10, ka59 0 (needs 'push the other block across the
  pit': a pushing planner, not a goal reading).
STILL MISSING (next): pushing as a planned relation (ka59); the
'select then place' protocol as a mode chain (re86 L2's three pieces,
r11l L2); relation classes not yet read: equal-to-reference, mapping,
sequence, count (sb26, cd82, tn36, g50t, wa30, tr87); the budget
reader; the organ's exploration cost on levels the archetypes solve
(a cheaper first pass, or archetype-first when one has confirmed on
this game).
Repo snapshot: arc3-repo-s23e.zip; bundle my_agent_v31.py (verified to
reproduce the module results on vc33 su15 re86 r11l s5i5 ft09).

# RECALIBRATION 2 (2026-09-06) — THE CORE REWRITE; THE UPLOAD GATE

Tushar's decision after the session-23e reading (22/183 levels on the
24 public games, no game completed, v31 projected 0.26 < 0.30):
the apparatus does not beat the games it has studied because its
learners were grown on the wrong substrate — a cell-and-colour map —
while the games are written in objects and relations. Two binding rules:

R1. REWRITE THE CORE on an object-level substrate. Four layers, built in
    order, each measured before the next:
    (1) ENTITIES — components -> rigid groups (co-movement) -> classes;
        identity across frames by spatiotemporal continuity, dihedral
        matching and livery, never by colour alone; attributes: colours,
        shape, size, parts, position, selected.
    (2) RULES — effect templates over entities (translate / rotate /
        recolour / select / appear / vanish / grow / push), fitted to
        transitions under the standing evidence rules (adopt on one,
        verify by use, contradiction is a question, evidence in
        proportion to cost, the open hypothesis).
    (3) GOALS — the session-23e readings become a TEMPLATE LIBRARY
        searched over entity relations (align, cover, inside, equal,
        all-of, count, sequence, mapping), not hand-written detectors.
    (4) PLANNER + EXPERIMENTS — search over the induced program (the
        cell planner is the special case 'translate on passable');
        the next action is the one that best separates the live
        hypotheses when they disagree.
    The archetypes stay in the repo, ablated from the path as the new
    core covers their games; the score is allowed to dip meanwhile.

R2. THE UPLOAD GATE: no Kaggle submission until the agent's INDUCED
    MODEL matches ground truth on LEVEL 1 OF ALL 24 REAL GAMES — the
    entities and their roles, what every available action does, the
    win predicate — AND the belief audit on the never-inspected
    simulator holdout is clean. Measured by tools/mechanics_gate.py
    against machine-readable keys study/truth/<game>.json (authored
    from the session-23d ground-truth docs; the agent never sees them).
    Winning a level is not the gate; predicting the mechanics is.
    v31 stays staged and unspent.

## Session 24 (2026-09-06): THE GATE, THE KEYS, AND THE NEW CORE'S FIRST TWO LAYERS

1. THE ANSWER KEYS. study/TRUTH_SCHEMA.md fixes a closed vocabulary
   (roles, effects, conditions, goal kinds); study/truth/<game>.json for
   all 25 real games, level 1, authored by four agents from the cited
   engine studies and checked against the rendered first frame and a
   few stepped actions (never solved). The agent never reads them.
2. THE GATE. tools/mechanics_gate.py plays level 1 with the agent, reads
   its MODEL REPORT in the same vocabulary and scores entity recall
   (role + colour), rule recall (action + effect + colour acted on) and
   the goal (kind + mover + target). PASS = 0.8 / 0.8 / exact.
   BASELINE, the session-23e agent (wins 15 of 25 first levels):
     0/25 games predicted; entity recall 0.41, rule recall 0.27, goals
     right 3/25. The number Tushar's question pointed at, measured:
     the apparatus wins levels it cannot describe.
3. THE NEW CORE, layers 1-2 (agent/core/):
   entities.py — a frame is read into SPRITES (multi-colour rigid
     things): a component inside another's box and touching it is a
     part (a dot in a tip, a livery pixel, a cross's centre, a slider's
     brackets); touching small components forming a compact dense block
     are one sprite (a three-colour card); tiny same-colour pieces
     touching corner to corner are one drawing (a diamond of four dots);
     GROUND colours (>= 300 cells) never glue (a floor patch wedged
     against a block); thin lines glue only to a line they cross (the
     bars of a cross; ropes chain nothing); big components are terrain.
     The tracker keeps identity across motion, rotation (dihedral),
     livery change, growth, merging and splitting, and emits EVENTS:
     moved / rotated / recoloured / grew / shrank / appeared / vanished /
     merged / split. Parts that move apart are split for good.
   rules.py — every in-field event is a WITNESS for a template
     (action, condition -> effect of kind, magnitude); conditions:
     always, click_on(kind), click_near, selected (the sprite wore a
     livery), blocked (a key that moves the kind did nothing while
     something stood in its way -> that something is a wall). A livery
     change of <= 3 cells is SELECT; a jump to the click point is
     TELEPORT. Roles from the rules: controlled, mover, button,
     selector, wall, budget (a thing that changes at >= 60% of actions),
     floor (terrain). Nothing in either file knows a game.
   tools/core_probe.py — the new core alone, with a fixed survey policy
     (every key twice, every small thing and every panel part clicked
     once), scored by the gate's scorer:
     entity recall 0.41, rule recall 0.55 over 25 games after <= 40
     actions each — already double the old agent's rule recall, with
     no goal layer yet (markers, sockets, references and goals are what
     the missing entities mostly are). s5i5: entities 0.86, rules 1.00
     from a 21-action survey (both sliders, both arms, grow and shrink
     by 3, the wasted click). lp85 rules 1.00; ar25 0.85; r11l 0.80.
NEXT (in order): (3) goals on the sprite substrate — the session-23e
readings as a template library over sprite relations (same kind, socket,
constellation, HUD twin, unique-thing-reach, equal-to-reference, count)
giving marker/socket/reference/canvas roles and the goal kind; (4) the
push relation (co-motion of an adjacent kind in the key's direction),
slide, and the missing conditions (parity, at_edge) from the atom;
(5) an object-level planner over the induced rules and experiment
selection replacing the fixed survey; (6) the belief audit on the sim
holdout; then the gate, game by game, until 25/25.

## Session 24b (2026-09-06): THE METHOD CHANGES — surgical diagnosis on known levels

Tushar: even with every answer known, the program fails past level 1
of the same game; a rewrite done the old way lands in the same place.
BINDING: no big-bang replacement. The new substrate (agent/core) is
used only where a diagnosis names the old representation as the cause.
Every change is earned by a SURGICAL DIAGNOSIS on a failed level whose
answer we hold — same game, next level, where nothing is unknown —
classifying the first divergence between belief and truth as exactly
one of: A PRIORI gap (no representation for the concept), LEARNING gap
(the concept exists, the evidence was never gathered or applied),
INFERENCE gap (evidence in hand, wrong conclusion), EXECUTION gap
(right conclusion, wrong act); then the smallest change that closes
that one gap, verified on that level and the battery, logged with its
class. v31 is not submitted.

CASE 1 — s5i5 level 2 (L1 won in 23; L2 refuted at 20 explorations).
Truth: a four-link chain 12->10->11->14 with the tip on the last link;
resizing a link translates everything after it; a wall column and a
top bar between the tip and its diamond. Trace: the organ's first
hypothesis was RIGHT (tip -> diamond, carried from L1); clicks at
(57,42) and (57,12) each moved the tip by (0,3) — and the organ never
clicked them again. Divergence: the candidate filter "a colour that
moved under our own clicks is not a button" (born on r11l's ropes)
removed the bracket because the bracket wears the ARM's colour.
CLASS: learning gap — a rule stated over colours where the truth is
about things. FIX: a button is a THING that stayed put when clicked
(the clicked cluster's own centre), colour-blind; tiny moving-colour
fragments (rope pieces) still excluded. s5i5 L1 23, r11l 12 (kept),
vc33 25, su15 10, re86 41, lp85 3 levels — no regression.
RESULT: the tip now travels 24 cells toward the diamond (col 16 -> 40)
and stops at the wall column. SECOND DIVERGENCE: the goal lies past a
wall; the chain must be lifted (link 10) into the gap between the top
bar and the column, extended (link 11), then dropped (link 14) — 89
human actions. The greedy descent cannot conceive a detour; the old
planner searches only the cell map of a walking body. CLASS: inference
gap at planning — the program holds every effect it needs (four
buttons, +/-3, everything downstream translates) and cannot compose
them. WHAT CLOSES IT (generic, next): a planner that SEARCHES OVER THE
COMPOSITION OF INDUCED EFFECTS, simulating the frame (render the
predicted state, reject overlaps with things that refused before), the
same search for chains, pistons, sliders, pushers and walkers alike.
The failure atlas listed this as "greedy stalls" in vc33 L2 and ka59
L1 too: one gap, three games.

## Session 24c (2026-09-06): CASE 1 CLOSED — planning by composing induced effects; the scaling test

Built agent/core/planner.py and tools/core_play.py to test the diagnosis
of session 24b (s5i5 L2: an inference gap at planning) and to see
whether the fix scales. THE MECHANISM, stated game-free:
- effects are learned per COMPONENT (a base stays, a bar extends, a tip
  rides) as cell-level transforms: translate(v), extend(side, k),
  retract(side, k) — read from before/after cells with two cells of
  slack for a hole that rides a bar's end;
- a key's MODEL is the set of component transforms it caused; a
  refused retract teaches a minimum, a refused extend a maximum;
- a plan is a best-first search over key sequences in which every
  predicted state is RENDERED: a changed solid that would overlap
  another solid (unless the two were seen overlapping, or moved
  together) or leave the frame refuses the key; THE MARKER IS NOT A
  WALL (the target of the reading is where the mover goes); the
  heuristic is the mover's distance to the target in step units;
- the plan is executed predict-act-verify: the first contradiction
  (with the same two cells of slack) ends it, the observation joins
  the model, and the search runs again (six replans);
- several movers of one kind with several markers of one kind is one
  conjunctive goal (all at once, greedy nearest assignment).
Surgical fixes on the way, each from a named divergence: identity kept
through growth (a bar's old cells are still there); HUD rows excluded
from the model (a step bar's two halves change at every press); a
tight drawing inside a loose constellation is a socket of its own (two
diamonds six apart); the goal reader uses the detected playfield (a tip
at row 4 is not HUD).
RESULTS (new core alone: a 30-action survey, then the plan):
  s5i5 L1 WON in 35 (old core 23-29; human 20)
  s5i5 L2 WON in 56 — the level the whole old stack never cleared
    (human 89; a 26-key plan: extend the root link, lift the second
    into the gap between the two walls, extend the third, drop the
    fourth into the diamond)
  vc33 L1 WON in 10 (old core 17; human 7)
  vc33 L2 lost: the pump pair — an extend refused because its PARTNER
    is empty; the model has no conditional "refused while X is at its
    minimum"; a conjunctive precondition, not a geometry (next case)
  s5i5 L3 lost: two tips and two diamonds at once with rigid beams on
    the chains; the joint search exhausts (100 s) — either an invisible
    collision interior the frame does not show, or the joint space
    needs the assignment fixed first; (next case)
  ka59 L1, re86 L1, su15 L1, r11l L1, lp85 L1 lost in the new core:
    their effects are outside the planner's vocabulary — push (a
    neighbour moves along), select (a key changes which thing the
    others move), pull (a click NEAR a thing moves it toward the
    click), teleport (a thing jumps to the click), permute (a button
    rotates several things along a track). The old core still wins
    four of these five; the two cores are not yet joined.
VERDICT on "does it scale": the same planner, with no game knowledge,
took s5i5 from L1 to L2 and vc33 L1 to 10 actions; it stops exactly
where its effect vocabulary stops, and each stop is now a named
divergence with a class, not a mystery. The old agent was not touched:
no regression possible; the join (the new core as the planner behind
the old agent's goal organ) is the next surgical step, after the pump
precondition case.

## Session 24d (2026-09-06): FIDDLING — push, select, pull, teleport and the ground, learned by trying

Tushar: "push, select, pull, teleport and permute are not yet in the
planner's vocabulary. These are learned behaviour through fiddling for
humans... If something new is presented, the scientist should be
equipped with the right moves and fiddling around."  Built
agent/core/fiddle.py (the repertoire and the general effect language)
and tools/core_play2.py (fiddle -> experiments -> plan over a general
successor function).  Nothing in either names a game, a colour, a
coordinate or a threshold chosen against a game; every mechanism below
was earned by a named divergence on a known level and is stated
game-free.

THE REPERTOIRE (Fiddler.moves): press every key; click each of the
ten biggest things and press the keys again; click the parts of a
panel; click beside small things; click far from everything; press the
keys again.  Then, with the keys known: BUMP the marked thing into the
nearest other things (what does contact do?); when a push was found,
SHOVE the thing again where the ground beyond it is open (a fixed leap
or a slide?); when a relative law was found, click at two longer
offsets (its reach).  When no plan exists under what is known, NUDGE:
each mover, or a pusher standing behind it, pressed toward its marker —
the obvious thing, tried anyway; the record grows and the plan runs
again.

THE GENERAL EFFECT LANGUAGE (GeneralModel), all learned:
- fixed: key -> component transform (the session-24c KeyModel), with a
  contradicted effect never planned with; a click that a relative law
  explains, or that moved a mark (a selection), is not a button;
- selected: the mark that changes hands is the selection mark; the
  mark colour worn by what the keys move is the selection colour; a
  click on a thing of a kind that drew the mark selects it;
- bodies and the active one (no mark needed): things that never moved
  apart are one body; the body the keys move is the active one; a key
  that moves nothing but after which the keys move another body is a
  selector (the cycle) — built, not yet a win (re86: the crosses
  fragment when they overlap; identity churns);
- relative: a click at offset o from a thing that moved it by f(o):
  teleport (it lands on the click) or pull (clipped per axis to a cap),
  within a reach; at least two consistent observations (once is a
  transition frame);
- contact: a thing that moved by v into another that was displaced in
  v's direction; CONFIRMED as a push only by a press where the pusher
  moved and the other did not (a chain that always moves as one body
  is not a push); the other's displacement is what a push does to it,
  with whether the pusher moved too; the same law in every direction
  until contradicted; the pushed thing's mark rides with it;
- the ground: colours under an accepted move's destination against
  those under a refused one — a colour only ever refused is a wall (the
  letterbox); a thing a mover has lain over is FLAT (a frame on the
  floor is not a wall).
Identity (entities.py): NOTHING PASSES THROUGH ANOTHER and A DOT'S
COLOUR IS CHEAP — among the assignments of a small class of same-body
things, the smallest largest displacement, then the least total, then
the fewest emblem changes (a size change costs like a displacement;
"vanished" costs like a jump of 20); a merge needs half the old cells
inside a non-terrain host; a flash on an edge is not a divergence
(verification by centre, not by cell count).

RESULTS (tools/core_play2.py, the honest table, no old core in play
except to reach a level):
  s5i5 L1 WON 45 (joint)   s5i5 L2 WON 66   s5i5 L3 lost (unchanged)
  vc33 L1 WON 20           vc33 L2 lost (pump precondition, unchanged)
  su15 L1 WON 46 — by a teleport law learned from clicks beside the
    thing, then probed for reach; the plan is five teleports and one
    relative click into the ring
  ka59 L1 WON 78 — the first push win: selection by click, the shove
    (the pusher stays; the pushed thing flies 9, 12 or 15 — it slides
    until something stops it, which the model does not yet say), the
    letterbox as a wall, the frame as flat, the joint goal, and a
    nudge across the pit when no plan existed; human 11 actions
  ka59 L2, su15 L2, re86, r11l, tu93, ls20, lp85 L1 lost
  (five level wins on the new core alone, from four)
Mechanics gate unchanged (entity 0.41, rule 0.56): the gate reads the
RuleInducer, not the GeneralModel — the fiddling vocabulary is not yet
reported into it (next).

WHERE IT STOPS, each a named gap:
- su15 L2 (identity through a morph): a click near a dot makes the dot
  vanish and a block appear at the click — one thing that changed form
  as it jumped; the tracker has no vanish+appear = jump; the fiddling
  is precomputed from the first frame (a thing that appears is never
  fiddled with); the goal reader has no hypothesis (the HUD twin's
  colour is absent from the playfield — it is what the block will
  become).  A priori gap (object permanence through a morph) + a
  learning gap (adaptive fiddling) + a goal gap.
- ka59 L2 (three blocks of two sizes, a big socket): the joint search
  exhausts; the push record is inconsistent across sizes (the shove's
  length is "until stopped", not a number).  Inference gap: a slide law.
- re86 (implicit selection): the crosses overlap and their arms split
  and merge as components; bodies churn.  A priori gap: a body seen
  through occlusion.
- vc33 L2, s5i5 L3: unchanged (a conditional refusal; a joint search).
Old agent untouched; v31 remains unsubmitted; the gate stands.

## Session 24e (2026-09-06): the untuned scoreboard, the track, and where the goal layer is the wall

Rule adopted (Tushar's brittleness question; the 0.30 -> 0.21 history):
no blind run until the core wins on VISIBLE games it was not fitted to;
the five visible losers (r11l tu93 ls20 lp85 re86) are the scoreboard,
and a change may be aimed only at a named general gap with a driving
case from a game already won.  Constants get a note saying which level
set them.  Honest accounting below: the positional-effect mechanism
was diagnosed ON lp85 (it was the "permute" item on the vocabulary
list), so lp85 leaves the untuned set; four remain.

BUILT (each general, each with its driving case):
- IDENTITY THROUGH A MORPH (entities.py; su15 L2): a thing that
  vanished and a thing that appeared in one transition, within 24
  cells, are one thing that jumped and changed form (event 'morphed');
  a split needs a donor that is still there.
- ADAPTIVE FIDDLING (core_play2; su15 L2): a thing that appears or
  changes form under the fiddling is fiddled with in turn (click it,
  press the keys, click beside it; up to 12 extra moves).  su15 L2 now
  learns the block's teleport law from nothing but the dots; the level
  is still lost — the goal reader has no reading (the HUD twin's
  colour is what the block will BECOME; the ring is solid).
- UNDO (fiddle.py; su15): a key after which the frame (HUD aside) is
  exactly the frame before the previous action, seen after two
  DIFFERENT previous actions (an inverse key restores it after its
  opposite only — ka59's down-after-up was misread first), takes the
  last action back: never planned with, never fiddled twice.
- LOOK AGAIN: readings from the frame after fiddling join the first
  frame's (a block where dots were).
- POSITIONAL EFFECTS — THE TRACK (fiddle.py + core_play2; lp85 L1): a
  button's effect on a thing is recorded by SLOT (top-left); a key that
  moved several things by different vectors in one press is planned by
  slot, not by identity (a corner piece turns); a positional key is
  pressed until it teaches nothing new (a full ring closes in one
  press), then the track is planned on.  lp85 L1 WON in 35 (old core
  15; human ~30).  lp85 L2 lost: six buttons learned, slots contested
  where the lines cross the ring (a slot's answer depends on the key —
  that is allowed — and on which piece stands there when tracks share
  a cell), and the goal reader picks a HUD swatch as the mover.
- SET UP THE SHOT (core_play2; ka59): before shoving a thing toward
  its marker, walk it (it takes the keys when clicked) to the lane with
  the longest open run toward the marker.  ka59 L1 had become a coin
  flip on the fiddling trajectory; won again, 94 actions.

THE HONEST TABLE (new core alone): s5i5 L1 45, L2 66; vc33 L1 20;
su15 L1 46; ka59 L1 94; lp85 L1 35 — six level wins (from five).
UNTUNED SCOREBOARD: r11l tu93 ls20 re86 — 0/4.  Diagnoses (no fix):
  tu93: the goal reader names HUD counters as movers; the walls are
    between cells (edges), which a colour-of-the-ground law cannot say;
    goal gap + a position-keyed refusal memory.
  ls20: the goal reader names the whole floor as a marker; the plan
    exhausts in fog with no wall known; the old stack's cell navigation
    ("the unique small thing; walk to it") is not in the new core.
  r11l: the closest — the fiddling learns a relative law for the
    handle and executes two clicks toward the marks, then the
    46 phantom buttons (every click on the rope is a 'button') and
    misread marks swamp the plan; selection there is by LIVERY (the
    clicked handle recolours), which the mark-based selection does not
    read.
  re86: bodies fragment where the crosses overlap; identity churns.
READING: three of the four stop at the GOAL LAYER — the old reader
(session 23e) is what the new core plans from, and it misreads
navigation games and multi-swatch HUDs.  That is layer 3 of the
rewrite (R1), and the failure atlas already named goal reading as the
largest class on the hidden-set proxy.  The fiddling core has done what
it was for (six wins by learned push / select / teleport / track);
the next effort belongs to goals on the sprite substrate, with the
same rule: driving cases from won games, the four untuned as the
scoreboard, no blind run until at least two of them fall for free.

## Session 24f (2026-09-06): the goal layer, first cuts — and an honest scoreboard

Aim: goals on the sprite substrate, driving cases lp85 L2 (a HUD strip
read as the mover) and su15 L2 (no reading).  Built, all general:
- A STRIP IS A DISPLAY (entities.hud_strips): four or more small things
  of one shape in a line at an even pitch along a frame edge — a row of
  dashes, whatever colour each dash — are HUD; the goal reader never
  sees them (goal_scan hud_cells).
- REACH THE ODD THING (core_play2, look-again): once the fiddling has
  shown what the keys move, that thing goes to each thing that is the
  only one of its colour — the oldest reading, ranked below relations;
  a part rides its whole (part_host: a facing dot in a ring), so the
  ring is the mover, not its dot.
- WHAT NEVER MOVED IS A WEAK MOVER: after the repertoire, a reading
  whose mover's colour no action moved is demoted below those whose
  mover did.
- OBJECT PERMANENCE UNDER A WALKER (entities: covered / uncovered): a
  thing with twins whose cells are now all held by a distinguished
  thing that came from elsewhere is COVERED, not moved; when it shows
  again, same cells, same look, it is the same thing.  Three refinements
  were forced by the tuned wins, each general: a twin standing on
  ground that was EMPTY before means the thing moved there (s5i5's tip
  at the end of a grown bar); the coverer must have come from elsewhere
  (a track showing through a vacated slot covers nothing); a thing that
  rode a shift two others rode is not covered (lp85's full ring).
- A RING'S HOLE IS FILLED for effect reading (planner._filled): a ring
  whose dot changes side is the ring, translated.
- A WALL BETWEEN HERE AND THERE (fiddle.refused_at): a move the key
  makes elsewhere, refused from this place with nothing in the way and
  the mover unchanged (a flash is no verdict; an unselected twin's
  record is no intent), is remembered by place and direction.
- NO PLAN FROM HERE: keys that did nothing where they were tried are
  pressed once here (a key may work elsewhere).
- Experiments are not repeated: (thing, direction, lane) nudges tried
  once; four nudge rounds.
Regressions on the way, all caught by the battery and traced: ka59 lost
to false refusals recorded on flash frames and on the unselected twin;
lp85 lost to the covered rule twice (the track's vacated slots; the
shifted ring); s5i5 lost to the tip read as covered.  Fixed as above.

TABLE (new core alone): s5i5 L1 45, L2 66; vc33 L1 20; su15 L1 46;
ka59 L1 94; lp85 L1 35 — six, unchanged.  UNTUNED: r11l tu93 ls20
re86 0/4.  Confession: tu93 was diagnosed to drive the object-
permanence and reach-the-odd-thing work, so it is no longer untuned;
its reach reading is now right (ring -> the odd tile) and the plan
fails on a wrong effect vector learned under an identity mishap.
Three untuned games remain (r11l ls20 re86); no blind run.
COST: this session's compute went mostly to keeping the six wins while
the identity rules changed under them.  The finding worth keeping:
identity through occlusion is where a component substrate pays its
tax — every rule about covering needs three exceptions — which is the
argument for the sprite-level reading (rule A: a part inside a box) as
the unit the model plans over, with components only inside it.

## Session 25 (2026-09-10): THE SCENE — objects that persist, frames that render them

Tushar: "you know the answer already; either your comprehension has a
gap or our theory is incomplete."  The theory was incomplete in one
place: the tracker matched pieces of colour between frames and needed
an exception for every way a piece can hide.  agent/core/scene.py
replaces it with the theory a person uses: the world is a set of
OPAQUE OBJECTS on a background, each with a persistent colour map and
an identity, and a frame is what they render to.  Reading a frame:
- every object proposes the transform that best explains the pixels
  (vectors voted by its own cells; staying put wins a tie); the most
  certain are placed first and take their cells; an object whose best
  place is taken falls back to its next (a flashed block beside its
  twin);
- OPAQUE OCCLUSION IS A COMPLETE EXPLANATION: an object placed as moved
  whose old cells all lie under other MOVED, non-terrain objects did
  not move — it is hidden there (a tile under a walker); the thing
  with twins yields first; a shift three others rode is a shift;
- the re-read runs in Z-ORDER (small and movable on top, terrain under
  everything): what lies on top claims a changed cell as its own
  recolour (a flash, a mark that changed hands), what lies under keeps
  its colour there; ground never recolours — what shows on it is a
  thing;
- what no object explains: growth at an object's own edge (a bar, a
  piston — terrain included, since a wall that grows is a piston), a
  part that arrived inside an object's box (a tip at a bar's new
  end), or a new object, paired with what vanished nearby as a change
  of form; ground that shows where a mover stood is REVEALED, not
  grown (the previous exposure is amended, so no transform is read);
- objects are exposed to the rest of the core as their whole footprint
  (a ring with its facing dot is a square that translates) plus each
  non-body colour as a PART with an id anchored to the object
  (sid*1000+colour): a part rides its object under every transform in
  the planner, and rides the growing end of a bar; hidden objects are
  exposed too, tagged, so contact and flatness still see them.
Ground: a colour covering >= 300 cells is terrain in every piece.
Effects are read on the whole footprint; the experiment's subject is
frozen in approach plans (the bump had been "approaching" by moving
the target).

RESULTS (tools/core_play2.py, SCENE=1 default; SCENE=0 = the old
component tracker):
  s5i5 L1 WON 63     vc33 L1 WON 20     su15 L1 WON 50
  ka59 L1 WON 73 (deterministic now; the coin flip is gone)
  lp85 L1 WON 45     r11l L1 WON 99 — AN UNTUNED GAME FELL: nothing
    was aimed at r11l; the handle, its rope and its marks read as
    objects and parts, and the track mechanism carried it
  tu93 L1 lost but transformed: the ring is the mover, the odd tile
    the goal, and it walks the graph by trying and remembering
    refusals (reached two nodes from the exit); the level's metered
    half-steps remain unread
  s5i5 L2 lost (REGRESSION: the same 26-step plan the component
    tracker found now exhausts 200k states; the model is right —
    the search's branching changed under parts; open)
  ls20, re86 lost (unchanged)
Untuned scoreboard: r11l WON; ls20, re86 remain.  The tracker's own
unit trace (ka59 push, tu93 walk, lp85 ring, vc33 piston, su15 dots)
is the evidence the theory is the right one: every case the component
tracker needed an exception for is now the same rule.

## Session 25b (2026-09-10): time, things that stick, and pictures that must agree

Tushar: "ensure we understand the canvas well, whether things stick
with the player, shape transforms, a sense of time, shapes and
obstacles."  Built on the scene tracker, all general:
- A COUNTER (the game's clock): a thing that changes IN PLACE the same
  way under every key at nearly every press (a step bar, a life strip)
  is not an effect of any key; it leaves the key records and is kept
  with its rate (fiddle.counters).
- A TRIGGER and a DISPLAY: when the thing the keys move ARRIVES on
  another thing (overlaps it now, did not before) and in the same
  press a third thing changes without moving, the arrival changed it;
  recorded by the kind arrived on (triggers) and the thing changed
  (displays).
- Pictures: a picture on the ground or in a HUD box is its own thing
  (rule A no longer glues into ground-coloured hosts); ground never
  recolours — what shows on it is a thing.  A picture's SHAPE is read
  at unit pitch (a glyph drawn at 2x is the same glyph).
- THE DISPLAY PROTOCOL: a display with a trigger, and a static picture
  of its colour elsewhere whose unit-cell count is within half to
  double, is a reading — "change the display by arriving again until
  the two pictures agree, then bring yourself to the reference".
  Re-arrival is by stepping off and back with opposite keys.
- REACHED = STANDING OVER (the odd thing's centre inside the mover's
  box), not centre-to-centre within a cell; eight readings tried, not
  four.
- A track is worked by a BUTTON whose pieces take each other's places;
  clicks on empty ground or on things that then move are pulls, not
  tracks (the r11l rope had been read as sixteen tracks).
RESULTS: ls20 L1 WON in 45 (the display protocol: reached the tile by
plan, cycled the carried glyph, walked to the gate) — the second
untuned game to fall this week; the old stack had needed a
hand-written archetype for it.  s5i5 L1 63, vc33 20, su15 50, ka59 73,
lp85 45 held.  r11l L1 is a coin flip (won once at 99 by chance pulls;
its ropes are physics the vocabulary does not hold: an indirect
mover — the handle moves the diamond).  ls20 L2, re86, tu93 lost.
Untuned scoreboard since the rule: r11l (chance), ls20 (won) of the
original four; re86 remains; tu93 walks its graph but the level's
metered half-steps are unread.

## Session 25c (2026-09-11): the chain's retraction, the gate on the new core

- s5i5 L2 regression closed: under the scene tracker a link's end that
  its neighbour moved onto had been read as HIDDEN (the neighbour over
  it), so the retract was seen a press late as a double retract.  Rule:
  a THING partly under another has lost those cells (a bar retracted);
  only a thing wholly under another is hidden; ground under a thing is
  still there.  s5i5 L2 WON again (26-step plan, 102 actions).
- A chain's refusal is about its configuration, not one piece's place:
  position-keyed refusals are recorded only for a lone walker.
- A part with its own record under a key moves by it; a part without
  one rides its object.
TABLE (scene tracker): s5i5 L1 63, L2 102; vc33 20; su15 50; ka59 79;
lp85 45; ls20 45 — seven level wins, all stable; r11l, re86, tu93 lost.

THE GATE ON THE NEW CORE (tools/mechanics_gate.py --core; the report
comes from the general model natively: controlled / selector / mover /
wall / floor / budget / button / canvas / reference, translate / grow /
shrink / push / select / teleport / recolour, goal from the reading
that won): 1/25 PASS (tu93 — read right, not yet won), mean entity
recall 0.47, rule recall 0.32, goals right 5/25.  Old agent: 0/25,
0.41, 0.27, 3/25.  Honest reading: the new core describes what it
plays somewhat better and is nowhere near the gate; the misses are
one-third vocabulary the core lacks (equal / mapping / sequence /
count goals; canvas and reference roles; hazards), one-third reporting
mismatches (buttons named by the colour under the click; "nothing"
rules not emitted — now emitted), one-third games the fiddling never
gets a grip on (cd82 sb26 ft09 lf52 tr87: click protocols with a
reference and a canvas).

## Session 25d (2026-09-11): bin 1 — the report measures the core, not the translation

Tushar: "1/25 is not a good score; if we decided the skills, how are
they gaps; where is the algorithm wrong vs your implementation?"  The
25 gate reports sorted into three bins (recorded in the chat and here):
  BIN 1 implementation of the REPORT (seven games credited wrongly);
  BIN 2 skills decided and not ported to the new core: rotate as a key
    effect, conditioned effects (first press turns, second moves), the
    goal kinds equal / mapping / sequence / count / merge with the
    reference and canvas roles (eight games);
  BIN 3 the theory's false assumptions: one press = one settled world
    (things that fall, hunt, scroll after an action); objects are opaque
    (drawn crosses are not); effects are immediate and local (program-
    then-run protocols; ropes and mirrors move the target through an
    intermediary); passability by support, not colour.  And beneath all
    three: the fiddling is a fixed repertoire, not experiment design.
Bin 1 closed this session (each a report rule, not a core change):
buttons named by every thing whose box holds the click, not the colour
under it; a mover's grow/shrink needs a repeated record (a big thing
something passed over is not a mover); "nothing" rules emitted for
silent keys and for keys with refusals; a reading with the controlled
thing as mover is REACH; marks that frame a spot the mover fits are a
SOCKET (inside), marks the mover covers are covered; the target of an
inside reading is both marker and socket; hud things and bars reported;
kinds remembered for things no longer in view; and one core fix found
by the report — the ground is taught only by the SELECTED thing's own
moves (a shoved thing, a misread wall "moving" 16 up, had taught the
letterbox colour as floor).
GATE (new core, all 25 first levels): 3/25 PASS (ka59, sc25, tu93);
mean entity recall 0.56, rule recall 0.60, goals right 8/25.
Before bin 1: 1/25, 0.47, 0.32, 5/25.  Old agent: 0/25, 0.41, 0.27, 3.
Level wins unchanged (seven).  Next: bin 2, starting with the equal
reading and a paint protocol (a canvas and a reference; actions that
recolour the canvas by place and by the controlled thing's colour),
driving case cd82 with ft09 and tr87 as the scoreboard.

## Session 25e (2026-09-11): bin 2 begins — the equal reading and the paint protocol

Driving case cd82 (a canvas, a reference picture, a basket that walks a
ring of slots around the canvas, a pour key that fills the canvas half
nearest the basket in the basket's colour, two swatches that hand the
basket a colour).  What the core now knows, stated generally:
- A PAINT (fiddle.py): a key (not a click) under which a thing took new
  colours on the same cells while nothing moved — recorded with the
  cells and the colour.  A COLOUR CLICK: a click on a small thing after
  which ANOTHER thing took one of the clicked thing's colours on the
  same cells — a palette.
- THE EQUAL READING (core_play2.py paint_protocol): a painted thing of
  at least 16 cells is a CANVAS; a REFERENCE is a filled rectangle of
  the canvas's dims (or an integer multiple: whole-block rescaling) made
  of touching things that never changed, away from the canvas, with the
  most colours — and it must show something the canvas lacks (a picture
  the canvas already equals, or a one-colour picture in a colour the
  canvas wears, is no goal).  The goal is canvas == reference, cell for
  cell.
- WHERE the paint lands is learned by asking: the movement keys walk
  the controlled thing over its PLACES — a graph whose nodes are
  signatures (the cells the movement keys ever changed, occupied or
  not) and whose edges are presses; the paint key is tried once at each
  new place in the colour the mismatched cells want most, and what it
  taught is kept as IN / OUT / UNKNOWN cells (a cell that already wore
  the colour teaches nothing); a place whose unknown cells hold most of
  what is wanted is tested again in the other colour before a new place
  is walked to; the prior that keys come in inverse pairs predicts the
  way back where only the way there was walked.
- THE COLOUR IN HAND is a state: set by a colour click, shown by a
  paint; the thing that holds it (whose identity a turning basket
  loses) only seeds it.
- THE GREEDY PLAN: while canvas != reference, the place and colour that
  fix the most known cells for the least known damage (unknown cells
  are neither); every paint verifies and corrects the record; the
  game's clock (a counter's cells per press) bounds the asking, and a
  wound-back clock means the level restarted.
- Report: canvas / reference / controlled (the colours the movement
  keys shifted) / button (palette hosts); paint = recolour of the
  canvas; movement keys = translate of the controlled colours; a
  palette click = recolour of the colour-holder; goal equal only once a
  planned paint brought the canvas nearer (a failed protocol asserts no
  goal — sc25's toggled grid cells had claimed one).
cd82 L1 WON by the new core (71–85 actions; was lost at 81); gate cd82:
entities 0.83, rules 1.00, goal equal — PASS.
FOUND ON THE WAY (each general, each earned by a diagnosis):
- A silent click hands the keys to nothing: the Fiddler's keys queued
  "after a click" are "if" moves, skipped when the click changed nothing
  but the clock (the fiddle was spending a third of cd82's budget on
  them; every battery win got cheaper: s5i5 63/102 -> 45/66, su15 50 ->
  40, ka59 79 -> 58).
- Every key twice before anything else: a level's first press may be
  swallowed (sc25's tutorial demo), and one observation is no law.
- A part's id is anchored to its object: a part of an old thing is not
  a new thing (fiddle+ was chasing swatch centres).  Fiddle+ waits for
  the keys queued after a click, and does not re-click the thing the
  click just changed (cn04's second click on a piece deselected it).
- Click a thing on its own cell nearest its centre: a hollow thing's
  centre is not on it (cn04's pieces were never clicked).
- A pressed button's feedback is not the clock: the thing under a click
  does not count toward being a counter (lp85's flashing arrow had been
  read as a budget bar).
- The report reads the level's last frame, not the next level's first
  (a win loads the next level; s5i5's buttons were named in L2's
  colours).
- A track's button: the thing under a positional click, translating
  every kind it moved by slot (lp85 PASS: entities 0.71 -> 1.00).
- ROTATE as a key effect (planner.effect_of): the same shape a quarter-
  turn on, wherever it came to rest (the clamp's business, not the
  law's); parts turn with their thing in the successor function; the
  report says rotate (cn04: ACTION5 rotate credited; rules 0.60 -> 0.80).
- A rule names every colour a thing wore (a thing's colour may be a
  state: cn04's selected piece is black, deselected 12).
- Selection shown by the thing's own colour: a click that recoloured
  the thing under it, after which the keys move that thing, selected it
  (cn04: ACTION6 select credited).
GATE (new core, all 25 first levels): 5/25 PASS (ka59, tu93, lp85, s5i5,
cd82); mean entity recall 0.60, rule recall 0.65, goals right 9/25.
Before this session: 3/25, 0.56, 0.60, 8/25.  Moved: cd82 0.50/0.33/no
-> 0.83/1.00/yes; lp85 0.71 -> 1.00 entities; s5i5 0.71/0.67 ->
1.00/0.83; cn04 rules 0.60 -> 0.80; r11l 0.57/0.60/no -> 0.71/0.80/yes;
vc33 0.50/0.40 -> 0.67/0.60; lf52 entities 0 -> 0.2.  Down: sc25
1.00/0.80/yes -> 0.71/0.70/no (its pass was flattering — the ACTION2/3
translate credits were stray readings the leaner fiddle no longer
produces; the truth's vertical moves are impossible at L1 while big);
su15 entities 1.00 -> 0.75 (a hud strip read differently after the
economy); wa30 rules 0.27 -> 0.18.
TABLE (battery): s5i5 L1 45, L2 66; vc33 20; su15 40; ka59 58; lp85 53;
ls20 44; cd82 85 — eight level wins (cd82 new, tuned); r11l, re86, tu93
lost.  cd82 is TUNED now (driving case).  Untuned wins unchanged: ls20.
HAND-CHOSEN CONSTANTS this session: canvas >= 16 cells; a retest when
unknown cells hold >= 50% of the mismatch; exploration budget 70
presses and a clock floor of 2.5 presses; reference search over
touching clusters and single things.
NEXT (bin 2 continued): the constellation-to-constellation align
reading (cn04's stubs: same-kind parts riding two bodies, one brought
onto the other — the goal cn04 still lacks); mapping goals (ft09, tr87:
a canvas of tiles toggled by clicks or keys against a rule card);
sequence / count / merge; conditioned effects (wa30: a press turns
first, then moves).  Then bin 3.

## Session 25f (2026-09-11): the replay viewer (Tushar watches the core play)

Tushar chose the first road (keep going; blind run the day the second
untuned visible game falls) and asked to SEE the core fail and pass,
so that he can offer an abstraction rather than a solution.  Built:
- tools/core_play2.py records every action with the frame after it,
  the tracker's events, and a one-line NOTE of what the core was doing
  (fiddling: press / click / after-the-click; bump; shove test; track;
  paint: testing place / planned / retest / colour click; reading + plan
  length; plan step i/n; replanning; joint plan; walking; display
  protocol; nudge; silent keys).  tools/record_replays.py writes
  study/replays/<game>.json for the 25 real games (never the holdout).
- The viewer (artifact "Core Replays"): a game list with outcomes, the
  64x64 frame in the official palette with the click ringed, the action
  and the note at every step, a scrub bar coloured by stage (fiddling /
  experiment / reading / plan), the events the tracker noticed, and the
  core's final reading (goal, entities, rules) in the gate vocabulary.
ACCIDENTAL OBSERVATION (recorded so it is not forgotten or repeated):
the recorder's first run took every game id but sh*, so the new core
also played the sim tuning set and blind set 1 once; only the won/lost
line was seen, no replay was watched, and those files were deleted:
st* 17/33 first levels won, sx* 1/12, mz* 0/5, bt* 0/2.  The sealed
set 2 and the sh* holdout were not touched.  The recorder now takes
only games with a truth file.
Tushar's question — "have you solved the intelligence and are only
crystallizing it?" — answered in the chat: no; the vocabulary (objects,
effects, readings, protocols) is the crystallization of what I can say
about these games, and each session finds a word the core lacks; the
words come from looking at failures, which is exactly what the viewer
is for.

## Session 26 (2026-09-11): Tushar's two abstractions — TRANSFER and FIT

Tushar watched the replays and named two things, not solutions: (1)
the second level is lost with the same keys and the same gate — the
"understanding" did not transfer; (2) perception is of SHAPES, not
identical pixels: "fit the rope's end inside the ring", "fit the same
shape on the other", "match the two dots".  Both are now words.

TRANSFER (fiddle.py export/seed; core_play2 carry):
- What carries is everything known BY KIND, nothing known by name:
  key -> kind -> effect (well supported, uncontradicted); which keys
  move the selected thing; selector kinds and the selection colour;
  pushable, flat, floor and wall colours; undo and positional keys; the
  clock (its colours, box and size); relative click laws by kind; the
  paint key and the canvas kind; the display's kind, place (hud or
  not) and size with its trigger kinds (not the floor under the
  arrival, not the mover's own colours); the reading that won and how
  (display / paint / plan / joint).
- The next level opens by CONFIRMING, not relearning: every key once;
  a seeded law that held on some thing of its kind is dropped for the
  things of that kind that stayed (the kind alone does not make a
  mover); the repertoire is skipped when the keys confirm (kept, clicks
  only, when relative laws were carried); the protocol that won the
  last level is tried first.
- ls20 L2 now reaches the rotator by plan, cycles the glyph, and walks
  to the gate — refused: L2 adds two keys to collect first, a mechanic
  L1 never showed.  No L2 falls yet (cd82 L2 adds wedge pours and a
  third colour; ka59 L2 three blocks; lp85 L2 six buttons; vc33 L2 a
  pump precondition; su15 L2 a morph).  Transfer is measured by the
  actions saved before the new thing appears, not by wins yet.
FIT (goals.py): two new first-frame readings, ranked below sockets —
  FIT BY SHAPE: two single things of different colours, the same shape
  up to a turn or a flip: one goes onto the other (ar25's L's).
  FIT BY MARKS: like marks on one body and the same pattern (up to a
  turn) on another: the marks go onto the marks (cn04's stubs).  The
  marks are the mover; the target is the OTHER body's marks where they
  are NOW (re-resolved at every replan); verified as a set, not by
  name (like marks trade names, a docked mark takes the other's
  colour); when the fit is made and nothing happens, the other way
  round (a half turn) is tried once; when a plan arrives and nothing
  happens, one step off and back (some games judge on the next action).
PERCEPTION FOR FIT (scene.py, planner.py, fiddle.py):
- THE SAME SHAPE TURNED IS THE SAME THING: a quarter- or half-turn about
  the corner of its box is a placement the tracker tries when no shift
  explains 90% of a thing; it costs a tenth (a shift that explains as
  much wins); event 'turned'.
- WHAT TOUCHES A THING GOES WITH IT: a small thing touching (or a
  cell's gap from) a larger one whose reading is a shift or a turn is
  first tried where that shift or turn sends it (two identical stubs
  had traded places on every move); a rider is never reverted as
  hidden under its thing.
- RECOLOURED IN PLACE IS NOT VANISHED: a thing no shift explains whose
  whole footprint shows one other colour nobody else explains changed
  colour where it stands (a stub docked, a piece selected).
- A turn press says nothing about bodies (a body's pieces shift by
  different vectors and are still one thing); the selected BODY answers
  the keys — its comovers shift and turn with it in the successor
  function; a key under which the active body turned is a turning key
  for whichever body is selected (key_rotate); selection shown by a
  thing's own colour (selected_colour), with its records conditional
  on selection; a selectable block's mark has no law of its own (ka59
  had regressed on that).
- cn04: the docking plan (select B, walk, turn, walk) now executes
  exactly as predicted and both stub pairs dock (turn colour 3); the
  level does not advance — the hidden mark kinds mismatch; the half-
  turn retry follows, but the pieces then interpenetrate ("pieces pass
  through each other") and the tracker's opacity reading loses them.
  Interpenetration is the next perception word.
GATE (new core, all 25 first levels): 7/25 PASS (ka59, tu93, lp85,
s5i5, cd82, cn04, ar25); entity recall 0.63, rule recall 0.66, goals
right 11/25.  Before this session: 5/25, 0.60, 0.65, 9/25.  cn04 and
ar25 pass by the fit readings (goal align / cover), rotate as a
selected-body law, selection by colour.
BATTERY: L1 wins unchanged (8): s5i5 45/66, vc33 20, su15 41, ka59 54,
lp85 53, ls20 44, cd82 85; r11l re86 tu93 cn04 ar25 lost; every L2 lost.
REPLAYS: study/replays now hold L1 and, after a win, L2 (opened with
the carry); the viewer lists "<game> L2" rows.
HAND-CHOSEN CONSTANTS: turn placement costs 0.1 of the score and needs
90% of the cells; a rider within 2 cells, <= 60 cells, needs 90%; fit
readings at 8.3 (shape) / 8.6 (marks), -0.2 when turned; a mark is
<= 16 cells; the single-reading search 80k states (40k after an empty
one); a display seed needs the same kind, hud-ness and half-to-double
the size; a counter seed needs any colour it wore, half-to-double the
size and an overlapping box.
NEXT: interpenetration (things that pass through each other: cn04's
half-turn retry, ar25's mirror), then the collect reading for ls20 L2
(a small thing that vanishes when arrived on, counted by the HUD),
then the L2 scoreboard (cd82 wedges, ka59 three blocks, lp85 six
buttons) as the transfer's measure.

## Session 27 (2026-09-11): look first, then try — the fiddling ordered by the readings

Tushar: "intelligence is also using knowledge and applying it in
novel conditions; your fiddling seems random; look at the shapes
first and make sense of them; if there is a shape match, try it; when
the block can move, the hypothesis should be trying to reach there."
That is the bin-3 item "the fiddling is a fixed repertoire, not
experiment design", now built as the OPENING of every level:
- LOOK: the first-frame readings (sockets, fits, marks, twins) name
  candidate movers and where they go.
- WHO MOVES: every key once (once more only if nothing moved — a first
  press may be swallowed).
- TRY: a reading whose mover the core can already move (a key moved a
  thing of its colour; the keys move the selected thing and it is
  selectable; a click law of its kind; a track that moved its kind) is
  pursued at once — joint with its siblings when several of the kind
  go to several markers — with NO nudging and NO silent-key trials:
  "no way under what is known" hands over to the repertoire.  A
  reading whose mover does not answer the keys, in a clickable game,
  gets one targeted experiment: click the mover (a selection?), the
  keys if the click did anything, a click beside it (a pull?); if it
  answers now, it is tried.  Two readings at most.
- Only then the repertoire (its first round of keys already done), the
  bump / shove / track / look-again stages and the readings again.
- pursue() is now one function (the reading chase: joint, single,
  verify, replan, off-and-back, the half-turn retry) used by both the
  opening and the later pass.
RESULTS: su15 41 -> 14 actions (looked, clicked beside the blob, went);
ls20 44 -> 36; vc33 26; ka59 84 (the early joint try costs before the
push is learned — the game is a coin flip either way); s5i5 45 -> 63
and lp85 53 -> 65 (the targeted clicks on tips and tiles teach nothing
there — the honest price of trying the reading first).  Eight L1 wins
kept; every L2 still lost.  GATE 7/25 unchanged (a first version of
the opening had emptied the repertoire's key rounds and cost tu93's
pass — refusals map the walls; the later rounds are kept).
Bugs found on the way: socket readings ("fits frame") had been kept
out of joint goals by the fit-reading exclusion (a substring); fixed.

## Session 28 (2026-09-12): THE RELATION LANGUAGE — readings generated, not listed

Tushar: "What is intelligence, what is fluid intelligence?  There must
be solid learning and skills and then the ability to learn and apply
coherently.  Why are you not able to create a solution?"  Answer given
and recorded: the core had the loop (perceive, read, try, verify,
revise) but not a generative reading — every reading and every law was
a word I typed, and the compiled thing is narrower than the compiler.
Two paths offered: (1) make the reading layer generative — a language
of relations over the things on the screen, readings proposed for
every pair the relations support, scored by weighted cues, weights
adjusted by feedback; (2) a local vision-language model in the loop as
a hypothesis source, verified by the program.  Tushar: "GO WITH 1st.
Lets win."  Design in the project doc claude/relation-language-design.md.

BUILT (agent/core/relations.py; see the design doc for the full
language): things (clusters, constellations, mark groups, pictures),
eleven relations, readings for every supported pair with the hand
list's words, a prior over cues (noted), RelationWeights feedback (a
reading pursued to its end with nothing happening -0.5 on its cues;
the winning reading +0.5; carried across levels).  goal_scan is
replaced by read_frame_goals in play_level (first frame and look-again).
Measured before any press on the 25 first frames: the language
reproduces the hand list's top reading on 17/25 and improves ft09
(socket), sp80 (the odd thing, where the list had nothing), cd82 (a
picture pair); three cues had to be disciplined by looking: same shape
counts only when the two are the ONLY two of that shape and do not
touch (lp85's two arrows, cd82's two picture halves); a constellation
of the mover's colour outranks a lone dot of it (r11l); a shape fit's
base ranks below a constellation's.
ALSO THIS SESSION:
- COLLECT reading (play_level): several like small things of one
  colour, not the mover's, not hud — arrive on each nearest first; the
  first arrival tells (vanished or changed = collected; unchanged =
  refuted); after a collect the refused protocol is retried (the door's
  price paid); collect comes BEFORE the carried protocol when the keys
  have shown who moves (like small things on the floor are cheap and
  usually a door's price).  Report kind all_collected.
- Lives beside a step bar are hud (scene.py).
- LOOK BEFORE YOU STEP: a key's translate onto cells that NOW wear a
  wall colour is not pressed (the ground may change after the plan).
- A patch of wall colour is wall however small it is read (lifting).
- Readings whose mover the core can move rank above the rest; a reading
  whose mover never moved ranks below every one whose mover did (-10).
- The early opening re-checks every reading for actuation after each
  experiment: a click made for one reading may have actuated another's
  mover (lp85: the arrow readings' clicks turn the ring; the tile's
  constellation reading is then tried — L1 in 10 actions, from 53).
- A select step on a thing that is gone is skipped (ka59 L2 crash).
ls20 L2 DIAGNOSED (the level is now understood, not yet won): the bar
is a TIMER PER LIFE — 84 cells, four per action, 21 actions a life,
three lives; at zero the player dies and returns to the start with the
keys it collected kept.  The core collects both keys (two walks of ~20)
and reaches the rotator but the third life runs out before the gate;
the level needs each leg planned from the START within 21 actions.
The clock-per-life reading (a counter whose zero is a death that
resets position) is the next word; it is general (sp80, bp35 have
clocks too).
GATE (all 25 first levels): 7/25 PASS (ka59 tu93 lp85 s5i5 cd82 cn04
ar25), entity recall 0.62, rule recall 0.67, goals right 11/25 —
unchanged in count under the new reading layer, which was the first
thing to check (no regression from listed to generated).  r11l L1 won
once (54 actions) under the language's ordering and lost once — its
coin flip is unchanged.  BATTERY: eight L1 wins; lp85 10, su15 14,
vc33 32, ls20 44, s5i5 63, ka59 85, cd82 97; every L2 lost.
(Session 28, late) The reading order had depended on memory addresses
(ties among equal-scored readings broke by dict order of thing ids):
lp85 L1 was 10 actions in one run and 49 in the next.  Readings are now
scored in a fixed order (by box and colour); lp85 L1 is 10 actions
every run (human ~30).  The replay viewer is republished with this
session's runs (L1 and L2).

SESSION 29 — THE CLOCK PER LIFE; ls20 L2 WON (the first L2 of a level
whose first level needed a protocol).  Every change earned on a known
level and stated generally; no per-game constant.
REGRESSION CLOSED FIRST: ls20 L1 (37 -> lost 53 after the turn/rider
tracker work).  Three general faults under it:
- A display's picture is read from the FRAME inside the box of the near
  things, not from the tracker's remembered cells (pieces of a turning
  glyph trade cells; their memory lags the screen).
- The INK of a display is the sparser of its colours in the frame
  around it (a glyph on a board carries less paint than the board),
  counted on the frame: a tracked piece keeps cells recoloured under
  it.  Before: "the colour that is not the body", wrong once the body
  piece wore four stale board cells.
- A display's change under a key on a MINORITY of that key's presses
  (the glyph changing on the presses that arrived on the trigger) is a
  consequence, not the key's law: not planned with (it grew the display
  into its neighbour in the plan -> "grow-solid" -> no plan).  A change
  at (nearly) every press is a law, display or not (s5i5's arm retracts
  at every click of its button: kept — dropping it broke s5i5 L2).
- The target of a walk read as several pieces, and the small things
  touching it (the other colours of one tile), are not solids.
ls20 L1: WON in 28 (was 37; human 13).
THE CLOCK (fiddle.py, one reading, three words):
- A counter's VALUE is the count of the colour it spends (a bar that
  turns colour from one end) or its size when it shrinks; what it
  spends is what MOST presses spend (votes over (thing, colour, cells)
  — a refill or a reset is one press and not the rule).  The rate is
  re-learned per level (ls20 L2 spends four cells a press, L1 two).
  The clock is found again by place and colour when its identity churns
  (a bar that empties is read anew when it refills).
- A REFILL: a thing whose arrival raised the clock's value; the visit
  uses it up.  Prior: the clock's own colour on the floor is a refill
  (ls20's rings wear the bar's colour) — used only when no plan fits
  the clock without it.
- A DEATH: the clock at (or under) one press refilled and the
  controlled thing moved other than as the key moves it: a life spent;
  nothing about that press is a law of the key; the thing's new place
  is the spawn.  The display protocol's visit record starts over after
  a death (the picture is put back).
- The clock's colour on the floor is not a collectible (collecting the
  rings early killed the L2 runs).
PLANNING WITHIN THE CLOCK (core_play2.make_successors / _walk_over):
the plan state carries the clock's cells; every action spends the
rate; arriving on a refill fills it and removes it; a state under zero
is not a successor.  A walk is planned first within the clock with the
proven refills, then with the prior refills, and only then ignoring the
clock (a death puts the mover back; the walk is replanned from there —
the executor now compares every step with the plan's predicted place
and replans when the mover is elsewhere).  A walk may be asked to
arrive with `need` presses still on the clock (the visits to come).
THE VISIT LAW (display protocol): from the pictures after successive
visits, the rigid motion one visit applies (turn, mirror) is inferred;
the number of visits the reference needs is then known (or known to be
never — that trigger is left alone), and the walk to the trigger is
asked to arrive with the clock for those visits.
ls20 L2: WON in 67 (display) — walk to the rotator within the clock,
die once when the ring is too far (put back, replanned from the start
through the ring: refill learned), three visits, the gate through the
second ring.  Human ~30 in 3 lives.
CARRY FIXES found on the way: a carried law is PROVISIONAL (support 0)
— the first press that does something else replaces it, a refusal
stays a refusal (the display's 'extend' carried onto the player's
colour had contradicted the player's own law; the level was lost in 4
actions); the export carries the best-supported law per kind, displays
excluded; a display is seeded by PLACE as well as kind, hud-ness and
size (a panel is where it was; the player of the panel's colour is not
it).  A refused CLICK is remembered by place like a refused key, for
every thing the step was to move (s5i5 L2 had repeated one refused
click eight times).
(Session 29, continued) ls20 L3 WON (219 actions; three levels of one
game for the first time).  The third level changed the rules again —
the tile by the palette cycles the carried glyph's COLOUR, the rotator
is elsewhere, the gate glyph is another colour than the display's, a
strip at the room's edge is a TELEPORTER, the clock is tighter — and
each new rule was met with a general word:
- THE DISPLAY'S STATE is a shape in an ink (read from the frame inside
  the display's box; the board is what surrounds the box, the ink the
  commonest other colour); agreement means shape AND ink.  The
  reference is a picture on a board of the display's board colour
  first, of any ink (the shape is the picture; ls20 L3 draws the gate's
  glyph in blue and the display's in orange).
- TILES AND THEIR LAWS: a trigger is a tile (the small things touching
  at one place, less the mover's own colours); each tile has a law on
  the display's state, learned by visiting it — what it does to the
  shape (a rigid motion, or nothing) and to the ink (ink -> ink).  The
  visits are then a SEARCH over states: the shortest sequence of tiles
  whose laws reach the reference; a tile never visited is visited once
  to learn (nearest first); a tile no walk reaches is left; a visit
  that crossed the tile twice (a walk through it and back) is read as
  two applications (arrivals counted from the trail).  The visits and
  the trip after them must fit the clock: the trip from the tile is
  asked of the planner with the refills it knows; else a refill first
  when one is within reach; the refill nearest the reference is kept
  for the way out.  The tile [0,1] "turns the picture rot1"; the tile
  [0,8,14] "leaves the shape, ink 12 -> 9".
- A PORTAL: a press that moved the controlled thing not by its law —
  from that place the press lands it elsewhere; remembered by place and
  key, planned with (ls20 L3's strip).
- IDENTITY ACROSS A JUMP (scene.py): a thing gone from where it was and
  standing whole elsewhere is the same thing (a far-jump reading, taken
  before anything nearer claims those cells as growth; among whole
  readings the larger thing places first); a vanished thing whose exact
  cells lie inside a larger residual reading is split out of it, name
  kept; a body that jumped is paired as a group.  A thing whose every
  cell now wears a GROUND colour is gone (a ring used up shows floor).
  A small thing wholly inside a larger thing of its colour is absorbed.
  And in play: A BODY STAYS TOGETHER — a member the tracker placed far
  from the rest is swapped with the like thing beside the rest.
- The walk's goal is the whole BODY (the target's centre may lie under
  either half of the player); a walk stops the press the level is won
  (presses after a win had been spent on the next level's clock); a
  refused step in a walk is remembered by place; the reference's door,
  refused while the picture was wrong, is forgiven once it agrees.
- Report: a display protocol won by walking is a REACH goal (mover ->
  reference); lives beside a bar are budget; a display is hud; the
  thing whose arrival changes it is a tool; the visit law is a rotate
  rule under adjacent(tool).  ls20 L1 now ent 1.00 rules 0.75 goal OK.
(Session 29, late — Tushar: "if something has worked, the approach
should be directed"; "ar25 still not doing anything as per my
guidance?")
ar25 L1 WON in 20 actions (human 15).  The reading was already his
("cover: fit the same shape onto C11") and the plan brought the mirror
L exactly onto the target L — then did not call it arrived: the
target's centre was its BOX centre, the mover's its cells' CENTROID,
and for an L they differ by 2.4 cells (tolerance 1.0); the fit sat
there unrecognised and the core went back to fiddling.  General fix:
a shape target's centre is its centroid (like with like), and a cover
reading is satisfied when the mover's cells lie on the target's.
DIRECTED CONFIRMATION: with a carried level the core had pressed every
key to confirm what the last level taught; now it presses until one
press moves something (that says who moves) and the other carried
laws are verified by use — the executor replans at the first surprise.
ls20 L2 123 -> 45, L3 219 -> 63 (no deaths spent).  Earlier in the
session the tile-learning likewise stopped exploring a tile once its
law was known (never-visited tiles first).
Also: a wall clock per level (LEVEL_SECONDS, 600 s, hand-chosen) —
s5i5 L3 had planned without end.
GATE 7/25 (ar25 fell to rules 0.77 — it plays 20 actions now and sees
fewer refusals and no click; the win is the point), entity 0.67, goals
13/25.  BATTERY: L1 wins s5i5 63, vc33 32, su15 14, ka59 83, lp85 10,
cd82 97, ls20 28, ar25 20; L2+: s5i5 66, ls20 45 / 63.  ar25 L2 lost
in 20 (no plan: the axis moves at L2 — next diagnosis).
(Session 29, later — Tushar: "why does ar25 L2 lose? perception and
hypothesis and conclusions wrong again?")  ar25 L2 WON in 24 actions.
The hypothesis was right and so was the mover; the holes were in
perception and in the conclusion machinery, each closed generally:
- A MARK MAY BE MANY DOTS: the selection is drawn as a black dot at the
  centre of every cell of the selected thing (five for the shape,
  twenty-one for the axis), the unselected thing showing holes.  The
  mark-reader took marks of <= 3 cells only; a pattern of lone dots,
  none touching another, inside a host at least four times its size,
  is a mark now (fiddle.livery_host).
- A DOT SHOWING THROUGH A MOVER IS A HOLE, NOT A CHANGE OF SHAPE: the
  target's covered cells show a blue dot through the yellow; the mirror
  (72 cells) was over the 64-cell limit of hole-filling and read as a
  morph, its law dropped mid-plan.  Limit 400.
- LAWS PER SELECTION (planner.KeyModel.ctx): the same key moves the
  selected thing and what follows it — the mirror by 6 when the axis
  is selected, by 3 the other way when the shape is; up and down only
  with the shape.  Laws are recorded per (key, selected thing) as well
  as per key; the plan asks with its state's selection; a key never
  pressed under the present selection, when its laws under another
  differ from its laws over all, is UNKNOWN there — pressed once to
  learn (before the silent keys), not guessed.
- A CARRIED LAW REFUSED TWICE and never seen to hold is no law here
  (dropped; the plan stops pressing it and looks for another way).
- A PLAN THAT ENDS SHORT OF ITS GOAL replans (it had declared
  "arrived" when its last steps were refused at the edge).
- UNDO: a key that restores the world as it was before the last press
  that CHANGED it (a silent press between does not count; the HUD —
  the energy column — is masked cell by cell) after two different such
  presses; the votes carry across levels (ar25 L1 gave one, L2 the
  second).  ACTION7 is then no move for the plan.
- A REFUSAL IS REMEMBERED FOR THE MOVER'S OWN BODY only (its host,
  what moves with it) — a wrong law's other victims are not refusals
  (the undo's misprediction had barred the axis from moving).
- A retry from "here" counts the selection as part of here.
(Tushar, mid-run: "what changes in your movement is a full object —
even if the same thing changes colour or is moving continuously —
consider this as the starting theory; reject it only if you find it
repeated[ly contradicted]".)  Adopted as the tracker's stated prior
for identity: a thing gone from its place and standing whole elsewhere
is the same thing — the same colour and shape first, failing that the
same shape in another colour (moved AND recoloured is still the thing)
— held until contradicted.  Regression battery after: all wins kept
(gate 7/25, goals 13/25; L1 wins s5i5 vc33 su15 ka59 lp85 cd82 ls20
ar25; L2+ s5i5, ls20 L2/L3, ar25 L2).  The remaining identity rules
(far jump before growth, split out of a larger reading, a body as a
group, absorbed pieces, gone-to-ground) are cases of that one prior.
ar25 L3 DIAGNOSED, not won (87 actions): the axis is horizontal now and
selected; two grey shapes sit above it, their blue targets above and
below; NO yellow image is on screen at the start — the reflections of
the shapes across the axis fall below the frame.  The carried reading
("cover: the yellow image fits the blue shape") has no mover to name;
the core falls back to the repertoire and to a nonsense joint goal
(grey shape to grey shape "same-colour marker").  What L3 needs is a
DERIVED THING: the yellow image IS the grey shape reflected across the
axis — a geometric law verifiable on every frame of L1 and L2 (image
cells == reflect(shape cells, axis line)) and then usable to predict
where an image will be for any placement of shape and axis, on screen
or off.  A law between three things (a relation of the second
language), not a per-key effect; the next word if ar25 is pursued.

SESSION 30 — "DO BOTH": laws between things, and the perception-net
experiment.
1. LAWS BETWEEN THINGS (agent/core/thinglaws.py, the third channel of
learning).  The key records say what a press does to a thing; the
frames also say what holds BETWEEN things press after press.  A small
fixed set of relations is tested on every frame — MIRROR (b is a
mirrored across a line thing L), COPY (b is a translated twin of a),
keyed by kinds so they carry across levels; a relation that held on
>= 3 frames with none against it is a law.  A law DERIVES a thing
that is not on the screen: ar25's third level puts the yellow images
below the frame; the law "C4 is C5 mirrored across C10" (learned on
L1+L2, 41 frames) derives them (cells off the frame allowed), the
readings name them ("cover: the image fits the same shape C11" — a
derived thing fits its exact shape whatever else is congruent), and
the planner recomputes them from their sources after every action
(a derived thing is not moved by any law and is in nobody's way; the
line it is mirrored across is a reference, not a wall).  Holes are
filled before mirroring (an unselected shape shows holes for its
dots); a line may be read in pieces where another thing crosses it.
ar25 L3 is NOT yet won: the axis read in two pieces both wearing the
mark, the fragmentation of selection contexts, and the search size
(three selectable things, two derived) leave the plan at 8 cells
from the fit.  Diagnosed to that point; the machinery is general.
Fixes found on the way: a bar has laws only when LIVE (selected, or
a mirror's axis) — the clock's parts had got extend/retract laws and
blocked every plan; selection contexts only once the keys have shown
a mark to be a selection (a ring's centre is no mark — ls20 L2 had
lost to fragmented contexts); a carried clock rate is replaced by the
first press of the level; the selection context is the BODY of the
marked thing.
2. THE PERCEPTION-NET EXPERIMENT (tools/perception_net.py).  Can a
small net learn a thing's ROLE from how it looks?  Data: 1114 frames
of the 25 first levels, each pixel labelled with the role of its
colour in that game's truth (controlled / wall / floor / hud / goal /
interact / hazard).  A 60k-parameter dilated CNN, 5 folds of 5 games
held out, the eight symmetries as augmentation.  RESULT: on held-out
games the net names 56 of 508 objects' roles (11%); the hand rules
name 85 (17%) while claiming only hud/wall/floor.  Goal things 0/155,
controlled 20/107, walls 1/19.  Twenty-five games do not teach what a
mover or a target LOOKS like — because they do not look like
anything in particular: a role is a convention of the game, learned
by pressing and watching, which is what the core does.  The negative
result is the finding: the press-and-see channel is the right one for
roles; a net could still serve LOWER perception (one object or two,
a mark or a hole) where labels are consistent across games — no such
labels exist yet.  Nothing of the net is wired into play.
REGRESSION after both: gate 7/25 (entity 0.66, rules 0.68, goals
13/25); battery: L1 wins s5i5 63, vc33 32, su15 14, ka59 83, lp85 10,
cd82 97, ls20 28, ar25 20; L2+ s5i5 66, ls20 45 / 63, ar25 24.  All
kept.

(Session 30, continued — Tushar: "you know every answer to all
levels... what skills are we missing?  ... give me an artifact like
core replays on winning set vs what we lack, i will teach you how i
think in pattern and you can program it.")
1. THE SKILLS ATLAS (artifact "Skills Atlas", /home/claude/atlas;
project doc claude/skills-atlas.md).  For each of the 25 games: the
human key (goal kind and the winning moves, from study/truth), what
the core does today (gate + battery), and the skill it lacks; the
skills ranked by the games they unlock.  The ranking: goal kinds
beyond place (sequence / mapping / count / merge / program: 9 games);
directed credit — what changed, by how much, toward what, INCLUDING
a change of state, a change far from the click, and a change that
shows only at a commit (9); effect vocabulary — toggle, delete,
slide, attach, cycle, swap (9); which body answers the keys — a key
that cycles control, a key that grabs (8); laws between things beyond
mirror — midpoint, stream, cargo on a track, ghost (5); conditional
laws — while / when / only if (5); deferred effect / commit (5);
reading a legend or table as a rule (3, the one wholly missing
organ); perception of space — camera, growing body, size against a
gap (4); planning with other movers and time (6); click protocols
whose affordances appear (3); economy — stop fiddling when the credit
is settled (6).
2. tu93 L1 WON (28 actions; human 18) — the ninth first level, by
Tushar's principle "what changes in your movement is a full object":
the block's eye (a PART, sid host*1000+colour) had its own translate
law (-8 on up: the eye turning to the top side) and was planned as a
mover of its own, so two presses in it "hit a wall" and the search
died at 85 states.  Now a part whose object moves under the key RIDES
it (its own record is how it is drawn within the object, not a motion
to check against walls); a part moves by its own record only when its
object does not move under that key.  Also earned on tu93: a move
sweeps the cells it crosses (_sweep), the fourth direction is read
from three (complete_directions), ground learning counts only the
mover's body and what the press moved as occupants.
3. REGRESSION CAUGHT: ls20 L3 lost in 2 actions after the tu93 work.
Two causes, both general: (a) tiles walked over had become ground_ok
and the trigger export excluded ground kinds, so no tile trigger was
carried to L3 and the display protocol never started — a trigger is
a thing arrived on whether or not it is walked over (only terrain or
large things are the floor); (b) the portal press (10,9)->(5,34) was
swept as if it were a step, so the wall colour 4 became ground_ok and
blocking emptied — a move that is not the key's own step (a portal, a
shove) teaches nothing of the ground.  ls20 back to 28 / 45 / 63.
REGRESSION after all three: gate 7/25 (entity 0.66, rules 0.68,
goals 13/25); battery: L1 wins s5i5 63, vc33 32, su15 14, ka59 83,
lp85 10, cd82 97, ls20 28, ar25 20, tu93 28 (nine); L2+ s5i5 66,
ls20 45 / 63, ar25 24.  All kept.  Replay viewer at Version 8 (tu93
L1 / L2 recorded).

## Session 31 (2026-09-13): Tushar's bp35 pattern, programmed — bp35 L1 WON

Tushar's reading of bp35, step by step ("1) fiddle on green, it
converts to blue; 2-3) move right allowed, black does not move so a
wall; 4) blue is our background and there is a way upward; 5) the
player flies up and hits green; 6) convert the green; 7) the pink
door is game over; 8) take the learnings to the next levels; 9) a
red button flips up and down").  Each step became a general skill;
bp35 L1 WON in 26 actions (human minimum 16, baseline 21).  The tenth
first level; the first won by exploration.

1. ONE THING MOVED AND REDRAWN (scene.py).  The block with an eye
   was read as its 9-body TURNED (the eye swapped sides makes the
   C-shape its mirror) and the eye moved by its own vector; the laws
   were "rotate 2" and "translate -8".  Now: a thing of several
   colours whose whole footprint lands, shifted, on cells of its own
   colours is the thing moved (read before a turn); touching small
   things whose readings are poor are tried as ONE footprint shifted
   (a body and the eye beside it).  The tracker hands the whole
   motion to the laws (`moved_by`); `_eff` in fiddle.py prefers it
   over the cell-level transform when the size is the same.
   (Tushar: "what changes in your movement is a full object, even if
   the same thing changes colour").
2. A WALL'S DOT TWO CELLS FROM A WALKER IS NOT ITS RIDER: the rider
   rule needs the small thing to have left its place.
3. AN INVERSE KEY IS NOT AN UNDO: undo votes are keyed by the change
   taken back — the body's vectors — so left-then-right and
   undo-of-a-right are one vote; two DIFFERENT changes taken back
   make an undo.  No vote when the body did not move.
4. A CLOCKWORK: a kind that has moved under different keys and always
   by one vector moves on its own (the rising hunter wall): no key's
   law, not a body, and not evidence of a track (the positional
   reading is recomputed from evidence less the clockwork).
5. A THING DOES NOT AIM AT ITS OWN BODY: readings whose target is a
   member of the mover's body (the eye) are dropped; the marked
   thing elsewhere is no selection when the keys move an unmarked
   body.
6. THE GROUND IS WHAT STAYS: of the cells a move crosses, those the
   press left as they were, and what shows where the mover stood; a
   cell that changed with the move was the mover's own — its outline
   in the wall's colour is its LIVERY (offsets from the body's box,
   seen twice), no wall to it in the plan or the safety check.  A
   texture — lone cells of a colour strewn over the frame — has no
   laws.
7. A CLICK THAT REMOVES: the thing under a click whose cells then
   wear what surrounded it or the ground is gone — a removal law for
   its kind (carried across levels); the plan removes a thing of the
   kind with one click and slides on through it; what the removal
   clears beyond the box (a tile's edges) is learned as growth.  The
   fiddle does not click a second thing of a kind that answered so,
   nor the part points of a thing that is gone: the fiddle costs 7
   actions instead of 38.
8. A SLIDE (gravity): the controlled thing went where the key sends
   it and ON along one axis through ground until stopped — it slides
   that way after every move and after a removal beneath it.  One
   sight is the theory; a move by the law alone with ground open
   beyond is a sight against; when no way is found with the slide
   the way without it is walked, to see (ls20's portal looked like a
   slide along the floor: the next move refuted it).
9. THE CAMERA (scene.py): the one shift that carries the last frame
   onto this one (window agreement >= 0.85, explaining >= 0.6 of what
   staying put leaves unexplained, >= base + 0.1) is a SCROLL; it is
   offered to every thing as a candidate (with a bonus over staying
   put, which a periodic texture also fits); the fiddle reads the
   frame back in the old camera (the walker's motion is its motion
   in the world; only the walker's laws are read on a scroll press);
   every remembered place (refusals, portals, spawn, clock box,
   slots, readings' targets) shifts with the screen; the plan's
   target is re-derived after a scroll.
10. AN EDGE IS NOT A WALL: a lone line of the wall's colour, one cell
   thin, between two open cells is a drawing; the target's own
   surround in the wall's colour is its outline.  A thing other than
   the mover leaving the screen is gone from the state, no refusal.
11. EXPLORE: with nothing to aim at, the body walks to the farthest
   place it has not been (places kept in the world's frame); the
   first surprise (a slide, a scroll, a thing that comes into view)
   ends the leg, the readings are taken again, and the fresh ones
   are pursued.  The gem came into view on the third leg and "reach
   the odd thing" finished it.
Constants noted: scroll radius 24, agreement 0.85 / explained 0.6 /
margin 0.1, hint bonus 0.2, soft-shift 0.95 with exact >= 0.4,
texture = a colour with >= 20 lone cells, livery seen >= 2.
Still to do from Tushar's list: 7 (the hunter as a hazard: a level
lost on contact), 8-9 (L2: spikes; L4: the flipper — a click that
flips the slide's direction).
REGRESSION (three rounds; each caught real breakage, each fixed
generally): (a) a slider's twin skipped as "answered" — a click's
removal law now needs the ground to close over the thing (its cells
wear what surrounded it, or a ground colour); (b) an inverse key read
as an undo — votes keyed by the body's vectors, none when the body did
not move; (c) ls20's portal read as a slide — a slide must go THROUGH
ground, one vote per press, and a one-sight slide is tested (the way
without it is walked when none is found with it) and voted against
when the body moves by the law alone with ground open beyond; (d)
cd82's basket fragments fiddled with as new things — a piece that
appears in the colour of a thing that changed form this press is its
fragment; (e) s5i5's chain of arms read as a scroll — a scroll is
ONLY the global shift, needing >= 8% of the drawing changed, several
colours, and what stayed put not contradicting it (static cells that
fit staying put and not the shift <= 15%).
RESULT: gate 7/25 (entity 0.70, rules 0.69, goals 13/25); battery: L1
wins s5i5 63, vc33 32, su15 14, ka59 83, lp85 10, cd82 88, ls20 28,
ar25 20, tu93 28, bp35 26, and cn04 58 — ELEVEN; cn04 was never aimed
at (the turn now sits inside the align-by-marks plan: ACTION5 and
eleven moves).  L2+: s5i5 64, ls20 45 / 63, ar25 24.  All earlier
wins kept.  r11l L1 fell once (65) in the second round and not in the
third — noisy, not counted.  Replay viewer Version 9; Skills Atlas
Version 2.

## Session 31b (2026-09-13): Tushar's additions — toggles, the remembered board, hazards; honest deaths

Tushar: "orange boxes and markers: click a box, it becomes a path with
a marker; click the marker, it becomes a box again. Remember the board
and plan like a micromouse."  And wa30 / tr87 readings (below).
1. A REMOVAL THAT LEAVES A MARK: a click after which most of the thing's
   cells wear the ground and what remains of its colour is lone cells is
   a removal (the mark clicked is the thing again — `click_marks`).
2. THE BOARD REMEMBERED (`model.world`): every cell seen, in the world's
   frame (screen + camera), kept across scrolls — a micromouse's map.
   Used by exploration (places visited are in the world's frame); the
   plan over the remembered map beyond the screen is still to do.
3. A HAZARD: at a press that lost the level, what the body touched —
   or, when the body is gone from the frame, what stood at its last place
   and the first thing along its slide (a fall onto spikes) — and had
   never been touched and lived, is a hazard kind (`death_touch` vs
   `touched_alive`); the plan never enters a state where the body
   touches one; collect readings of a hazard kind are dropped.
4. A DEATH ENDS THE LEVEL, HONESTLY.  Until now a GAME_OVER was followed
   by RESET and play went on as if nothing had happened — the level
   restarted for free, and several "wins" (s5i5 63, ka59 83, cd82 97)
   had a death inside them.  Now a death ends the level at once
   (`_Died`, how="died") and the harness makes ONE more attempt from the
   first level, replaying the won levels from what they exported and
   carrying the death's lesson (hazards, laws).  s5i5 L1: died at 50
   (the fiddle spent 44 clicks of a 50 budget), then won in 45; bp35 L2:
   died on the spikes, then knew them and zig-zagged through the block
   rows to the open side — not yet won (search time).
5. Fixes on the way: many blocks of one colour are blocks, not floor
   (ground = a colour of >= 300 cells whose largest piece is >= 100); a
   thing lying under the body is flat only when it is still there, was
   not the thing just clicked, and the walker is the body (not a twin
   on a scroll press); a click after which the body slid along the
   slide's axis is the slide's doing, not the click's law on the body;
   removal actions are offered only within two steps of the body or
   along its slide; the body and its outline count as open in the
   edge test; the scroll is the shift that best explains what CHANGED
   (a periodic texture fits many shifts).
REGRESSION: gate 7/25 (entity 0.69, rules 0.68, goals 13/25); battery
L1: s5i5 died 50 / won 45, vc33 32, su15 14, ka59 83, lp85 10, cd82
88, ls20 28, ar25 20, tu93 28, cn04 58, bp35 25 — eleven; L2+: s5i5
64, ls20 45 / 63, ar25 24.  Deaths now seen for what they are: r11l
60, sp80 30, tu93 L2 4 (all budget or hazard game-overs).
NEXT (Tushar's two readings): wa30 — the blue squares are handles; a
press toward a crate turns the body first (a blocked press is a turn),
ACTION5 attaches the faced crate (its border turns black: a livery of
the held state), the crate moves with the body in any direction, ACTION5
drops; all three inside the pen (count).  tr87 — the top is a table of
pairs blue -> pink; each bottom blue glyph is a table glyph turned; the
pink under it must be the pair's pink turned the same way; up/down
cycle it, left/right move the bracket: glyphs up to a turn + the table
as a legend applied slot by slot.

## Session 31c (2026-09-13): Tushar's method — the deterrent, the working hypothesis, the stop — bp35 L2 WON

Tushar's replay of bp35 L2, in his words: he died once on the door, did
a full root-cause analysis and set a deterrent (what kills, what to
avoid, by looking ahead); then held a working hypothesis of "what
works" (only left/right act; a blue cell above means I travel up; the
gem was up, so up is the way) and acted on it in the most efficient way
until contradicted; adapted only what changed; fiddled only when the
situation was impossible.  "The program should be able to do this for
anything new too."  Programmed as general organs; every step earned by
a diagnosis on a known level:

1. THE WORKING HYPOTHESIS (carried level).  What won the last level is
   what wins this one: its goal colour is carried (`won[2]`, from the
   reading's t_color or its note); if a thing of that colour is in
   view the plan goes to it (prio 9); if not, it lies where it lay
   before — `won_dir` = sign(goal − start) in the world's frame — and
   the body explores that way, leg after leg (up to 40, 60 % of the
   level's time), by free ground first and by a removal only when no
   free way leads ON (progress along the direction decides between the
   free pass and the removal pass).  Nothing else is tried until this
   fails; then the old flow (fiddle, readings).
2. THE DETERRENT, ROOT-CAUSED.  A death's cause is looked for along
   BOTH rays: the key's move (what it walked into) and the slide from
   where the move put it (what it fell onto); the first thing met is a
   hazard kind.  Kin: a kind every thing of which touches a hazard
   (the blades over the spike's base) is of the hazard.  Look-ahead: a
   state is deadly not only when the body touches a hazard but when a
   hazard lies along its slide with nothing solid between — a drawn
   line is no stop; the body cannot stop short of it (`deadly()`).
3. THE LATTICE.  The slide moves the body by the keys' own step (six
   cells here), not cell by cell; a step anything blocks is not taken,
   so the body never comes to rest off the lattice (the old model slid
   it one row into a drawn line).
4. THE STOP.  With a slide known the body is at rest after every press:
   the wall-coloured cells just beyond it in the slide's direction held
   it — recorded in the world's frame (`model.stops`) and treated as
   wall by every plan, drawing or not.  (The edge rule stays: a
   one-thin black line between open sides is a drawing — the grid lines
   between bp35's boxes are passed through once the box is gone.)
5. THE SCREEN THAT FLEW.  A scroll of 36 rows (the body flying six
   cells) was missed: the overlapping window was "too small to speak"
   and the world of two grounds (black maze on blue) has no one
   background to lift.  Now: shifts within 24 all round and straight
   ones to 48; a window of a third of the screen that agrees ≥ 0.9
   counts; and when the background-lifted reading finds nothing, every
   cell is counted under stricter thresholds (frac ≥ 0.9, expl ≥ 0.9,
   ≥ 15 % changed).
6. WHAT IS WHOLE WHERE IT WAS, STAYS (tracker).  A thing found entire at
   its own place is placed before any thing that would move onto it —
   a vanished box among identical boxes no longer shuffles the row
   along (each box "moving" 12 onto its neighbour, the corner pixels
   cascading), which had polluted the click laws with translations.
   The `stays` flag precedes size in the placing order.
7. THE DETOUR.  When nothing leads on the way the goal lay — no free
   way and no removal with progress — the leg falls back to the plain
   farthest-unvisited exploration (Tushar: an impossible situation is
   where to look about), rather than stopping.
8. A HIDDEN THING IS NOT IN THE WAY.  The tracker's "hidden under
   another" reading is an inference, not a sight; a ghost of the body
   left under a box after a 36-row flight blocked every plan leftward.
   Hidden things are not solids for the planner until they show.
9. The retry carries `won_dir` too (it was computed only on the normal
   path), so the second attempt at a level explores the way the goal lay.
RESULT: bp35 L2 — dies once flying under the spikes (first sight,
exactly as Tushar died on the door), the retry carries the hazard,
reads the blades as kin, refuses the click that would fly it under
them, explores up through the box rows by removals (detouring when
the shaft's top leads nowhere), sees the gem and plans to it: WON in
46.  L1 25 both times.
REGRESSION: gate 7/25 (entity 0.70, rules 0.69, goals 13/25); battery
L1 unchanged — s5i5 died 50 / won 45, vc33 32, su15 14, ka59 83, lp85
10, cd82 88, ls20 28, ar25 20, tu93 28, cn04 58, bp35 25 — eleven; L2+:
s5i5 64, ls20 45 / 63, ar25 24, bp35 46 (new).  The replay recorder now
mirrors the harness's honest retry (the L2 replay shown is the second
attempt, the first kept beside it); the viewer's L2 entries fetch by
file (they were fetching "ar25 L2.json" — a bug since the L2 rows were
added).  OPEN: one recorder run of the bp35 retry lost in 13 (no plan)
where three others won in 46 — the explore's 4000-state horizon makes
the outcome sensitive to order; not yet root-caused.
CONSTANTS (hand-chosen, noted): legs ≤ 40 and ≤ 60 % of level time;
direction bias 2.0 per cell in `far()`; scroll fallback thresholds
0.9/0.9/0.15 and the third-of-screen window at 0.9 agreement; hazard-kin
exoneration when touched alive more than twice per thing.
NEXT: wa30 and tr87 readings (unchanged from 31b); the belief reset
(when a contradiction should overturn the working hypothesis, not patch
it); death places remembered in the world's frame; planning over the
remembered map beyond the screen (micromouse).

## Session 31d (2026-09-14): L3 — the toggle, the impossible situation, the belief reset (not yet won)

Tushar: "did you try L3? ... the reverse should also be self-fixed."
bp35 L3 tried; what it exposed, in order, each fixed generally:

1. THE IMPOSSIBLE SITUATION FIDDLES.  On a carried level no click was
   ever tried (the keys were known, so the fiddle list was empty); when
   no reading plans and nowhere leads on, a thing of a kind AND FORM
   never clicked is clicked (one per kind-and-form, nearest first, up
   to three rounds), and the readings and the exploration are tried
   again from what it taught.  L3's red block: a click turns it into a
   mark of itself (passable); the block beside it is then removed and
   the body walks through.
2. A REFUSAL LAPSES.  "A move refused from that place" was permanent;
   the block that refused it had become a mark and the plan still would
   not step there.  Refusals now remember what stood where the move
   would have gone (`refused_seen`) and stand only while it looks so.
3. THE TOGGLE'S OTHER WAY: A CLICK THAT RAISES.  A click on a mark after
   which a thing of its colour several times its size stands centred
   over it raises the block (`click_grows[kind] = (mark size, block
   dims)`); the planner offers such clicks (the mark being the cluster
   of the kind's small pieces within a block's span, the block rising
   centred on it as a solid of its own) within the body's reach — a
   ceiling to walk under where flying would meet the spikes.  A removal
   click is offered only to things at least half the size a click has
   been seen to take away (a mark of the kind is raised, not removed).
4. KINDS CONFLATE FORMS.  The same colour is a solid block and a
   passable mark: once the body had walked through the mark, the colour
   was "ground" (`ground_ok`) and "flat", and the raised block was walked
   and flown through in the plan.  Both are now by form: a thing of a
   ground colour that is also known solid in another form (removed by
   clicks, raised, refusing) is ground only when no bigger than what the
   body ever stood on (`ground_size`); a flat kind's thing is flat only
   up to twice what the body lay over (`flat_size`).
5. THE TRACKER AND A RAISED BLOCK.  The block rose inside black lines
   and was read as the black terrain growing (a multi-colour residual
   whose body colour was the wall's): what grows the terrain is now its
   own colour only, the other colours inside are a thing of their own;
   and a thing that grows into its small neighbours of its colour
   absorbs them (the mark's pixels, read apart, become the block).
6. THE BELIEF RESET (the reverse).  One sight of the body sliding the
   other way overturns the slide law whatever its votes, clears the
   stops, and records the flip with the kinds the body touched on the
   press before (`slide_flips`) — the likely cause; and the last three
   sights all against a slide drop it however old.  Not yet exercised:
   bp35's gravity flipper lies beyond L3.
STATE: L3 reaches the ceiling of raised blocks and walks under it to
the shaft, flies up (30 rows: the scroll reads it), and stalls in the
upper room after two more legs — the remembered world beyond the screen
is not yet planned over (micromouse), and I have not yet read L3's
upper layout well enough to say what the level asks there.  L1 25, L2
46 unchanged.
REGRESSION: gate 7/25 (entity 0.70, rules 0.68, goals 13/25); battery
L1 the same eleven (cd82 91); L2+: s5i5 64, ls20 45 / 63, ar25 24, bp35
46.
OPEN: planning over `model.world` (the map beyond the screen); the
raised block's identity through further re-reads (kinds by form is a
patch on a deeper need: a thing's FORM as part of its kind).

## Session 32 (2026-09-17): WHAT IT SAW — perception in the replay, for Tushar's review

Tushar: "tell me what you perceived, so I can see perception at runtime."
The workspace had been reclaimed; restored from arc3-repo-s35.zip
(python3.12 venv, `pip install "arc-agi>=0.9.1" pillow numpy` — the
PyPI arc-agi that a 3.11 venv resolves to is a different package).
Every replay frame now carries `seen`: every thing the tracker holds
(name, colour, size, box, tags) with the ROLE the model gives it at
that moment (body / hazard / removable / mark / ground / flat /
pushable / clockwork / wall / hidden / unread thing; terrain and hud
set aside), the model's standing beliefs (camera, slide, ground, walls,
removable, marks, hazards, flat, stops, refusals, the key laws), and
the remembered board with the camera rectangle.  The viewer draws the
things as boxes coloured by role with their identity numbers (identity
failures show as a number that jumps), names the thing under the
pointer, lists them with their roles, and shows the board beyond the
screen.  All 25 first levels and the won second and third levels were
re-recorded from the current code (11 first levels, 17 levels in all,
bp35 L2 46 after one death; s5i5 L3 recorded as lost).  Nothing else
was run: the review comes first.

## Session 33 (2026-09-20): cn04 L2 — the winning reading as a template, and what it took

Tushar: "ok. Just 1 level please. not full regression." Only cn04:1-2 was
run (no battery). His readings for ft09, g50t, ka59, lf52 are recorded in
the project doc claude/human-readings.md (not yet in the program).

The template organ works: the reading that won L1 ("marks onto like
marks, turned") is read on L2's things and pursued first, on every
pursuit, before joints and junk readings. What the level then exposed,
each fixed as a general rule (no game-specific constant):

- THE READING NAMES ITS MOVER BY PLACE, not by the selection (four pieces
  bear marks; `resolve_fit` keeps the host it found; the selected thing
  only breaks a tie). Marks read as PARTS (sid*1000+colour) of a piece
  are its marks too. A mark's host is what it MOVED WITH (bodies) before
  "the smaller adjacent thing".
- A LAW LEARNED ON KINDS IS PRESUMED OF LIKE-FORMED THINGS: the selectable
  FORM (the mark colour every selectable piece wore) is carried; a piece of
  a new colour wearing it is presumed selectable (`select_presumed`), the
  click tests it, a failed test refuses the kind. Re-presumed at every
  plan (a piece shows its rest colour only once another is selected).
- The turning key (`key_rotate`) is carried across levels: it is about
  the key and the selection, not a kind.
- SIDE MARKS RIDE THEIR PIECE in the planner (touching boxes are one
  apart — an off-by-one had silenced the rule); a mark the keys moved
  rides the piece the keys moved with it; a mark of a member moves once;
  a part whose record is dropped rides its piece; the piece and its marks
  turn about the box of the whole (as the game does).
- TRACKER: a stub turns with its host about the whole's box (either
  pivot tried); a rider of a turned piece is not a stayer; a rider that
  changed colour where it landed (a docked mark turns green) still rode;
  a hidden mark under a rider does not ride too; a thing whole where it
  was stays even when a part already explains its cells (no far jump);
  RECOLOURED WHERE IT STANDS — a thing whose footprint shows one other
  colour (the selected piece drawn in the selection colour, the other
  piece dropping to its rest colour) is the same thing, also under
  partial cover. Identity no longer follows the colour across the board.
- Docking by a PAIR of marks (`_marks_fit` needs two; goal and heuristic
  per mark, a merged part split into its marks); only FREE marks dock or
  are landed on (covered / recoloured ones are taken); a docked reading
  is DONE (the marks took another colour, also when the part was renamed
  by its new colour), the frame is re-read, and a done docking whose
  piece no longer wears a docked mark is open again.
- SOLIDITY IS LEARNED: `passable` (kind pairs seen to stand on each
  other's cells) is carried; pieces seen to pass a piece are presumed to
  pass the other pieces; the target's own mark cells are no obstacle.
- SELECTION WITHOUT A LIVERY: `selected_now` falls back to the thing the
  keys last moved (clicking it again lets it go — the plan no longer
  toggles the selected piece off).
- When the keys are confirmed from the last level but no click was ever
  learned, the fiddle keeps its clicks (a piece is chosen by clicking).

Result: cn04 L1 won (18 actions at best — the opening plan now finds the
docking directly); L2 reaches two of three dockings (the 9-piece onto the
green, the white onto the blue) and fails at the third: the green must
move with the 9-piece docked on it (the game does not bind docked pieces;
`bound` was tried and withdrawn), the docking order matters (a pad
should dock before what lands on it — the "hub first" ordering could not
be read from the reversible partial fits), and the level's ~100-action
budget is spent by the click fiddle when L1 was won without a click.
Open: docking order by dependency; a cheaper click fiddle; the third
docking. Snapshot s37 = this state.

## Session 34 (2026-09-20): cn04 L2 WON (L1 18, L2 67 — twice)

Tushar: "go ahead. implement please." What it took after Session 33, each
a general rule:
- THE CHEAP CLICK PROBE: keys known from the last level, no click law —
  click each large thing, press a known key and its opposite (it answers
  or not, and stands where it stood); no fiddle+ on things that "appear"
  during the probe. The selectable FORM is read on this level when the
  last taught none.
- Things that moved together by one vector, touching, are ONE BODY on that
  press (a piece and its marks that had never moved before) — so the click
  that recoloured it is confirmed as a selection.
- SELECTION IS A LAW CONTEXT also when shown by the thing's own colour or
  by which thing answers (`sel_ctx`, `sel_b` from `selected_now`); a key
  that moved the chosen thing is a key of the chosen thing at one sight;
  the other arrows are completed from the kind laws on the selected colour.
- AN OPPOSITE ARROW IS NOT AN UNDO: undo votes must name a change other
  than the exact reverse of the key's own translate; the change's
  signature is whatever moved, active or not.
- The template, when readable on the level, replaces the "reach the goal
  colour" opening. After a docking the world is read again and the chase
  restarts from the readings as they stand (a docking may undo another;
  stale docking readings are dropped and read afresh). A host with no
  free mark left is no mover (re-resolved by place).
- THE DOCKING LIVERY: the first docking teaches the colour a docked pair
  wears (3 here); afterwards a docking = the target's marks wearing it
  (counted on the target, since a mover may leave another docking behind)
  AND the mover's marks on them. Every docking made and the level going on
  → one pair is the wrong way round: turn a docked piece half round and
  dock again (only once no mark is left free). Not needed on L2.
- plan_general is bounded in time (90 s) as in states.
Open: L3 (grey masking: connectors only visible on the selected piece);
"kind = colour" still makes phantom readings between the selected colour
and the rest; the junk "reach the odd thing" readings still run when
nothing else can. Snapshot s38.

## Session 35 (2026-09-20): ft09 — THE LEGEND WITHIN (all six levels won: 4, 7, 14, 16, 21, 13)

Tushar's reading: "The middle square has a shape. There, the white is red
and gray is black." — the card in the middle of each panel is a miniature
of its ring: each cell stands for the tile in that direction, its colour
names the tile's. Read at two scales (the panel as one thing; the relation
between its centre and its periphery), the three consistent panels are the
examples and the framed one, which breaks the rule, is the puzzle.

agent/core/legend.py (no colour, size or place named): the tiles are the
one-colour patches of one size that cover most of the board, on a lattice
whose pitch is the nearest-pair distance; a legend is a box of that size
at a lattice point drawn in colours no tile wears (its centre apart), with
uniform k×k cells (k=3 for the eight neighbours); a tile with a pattern in
its own colour (a cross, a notch) is a tile, not a legend. The mapping is
glyph colour -> "same as the card's centre" / "other than it" (so it
carries to cards of another centre colour, and to three-colour palettes
where "other" means either other colour). It is learned from the panels
that are not untouched puzzles (tiles of one colour do not vote), carried
across levels as the working hypothesis (10 votes), and refined by the
level's own examples. A tile between two cards must satisfy both.

Acting: a click on a tile changes its colour — the first click shows what
a click on a tile of that LOOK changes (itself; or itself and its
neighbours for a cross or a notched tile), learned on the instance with
the most tiles around it; tiles that look alike act alike; the effects are
carried. Plain tiles: click each wrong tile until it satisfies its demands
(the palette cycles). Tiles that change their neighbours: the set of clicks
that makes exactly the needed change is solved over GF(2) (Gaussian
elimination), re-solved after each round from what the clicks were seen
to do. Human baselines 43/12/23/28/65/37; the core: 4/7/14/16/21/13.
Checked on every game's first frame: only ft09 shows a legend within.
Snapshot s39.

## Session 36 (2026-09-20/21): ka59 — SENT THERE BY A STRIKE (L2 won: 56 actions; human 109)

Tushar: "Ka59 is solvable. I already shared the solution on L2 and it
needed a plan" — the two big blocks may not enter the pit; each must be
walked to a launch and struck across it by another block, and the
striker itself routed there (across the pit, if need be, by a third).

What the level actually needed, found by surgical diagnosis on L2:
1. A MARK IS SMALL AGAINST ITS HOST (fiddle.livery_host): the 2×2 centre
   of the 6×6 block was no mark (marks were ≤3 cells), so the big block
   was never "selected" in the record, its refusals at the pit were
   charged to the last selected thing, and the pit never became a wall.
   Now a patch of up to a sixth of an enclosing host (≤16 cells) marks it.
2. WHAT A SHOVED THING SLID OVER (fiddle.observe): the sweep was taken
   along the nominal law's vector (18) on the after-frame — it counted
   the letterbox, the floor and the blocks themselves as "slid over", so
   a simulated shove never stopped.  Now: the way it ACTUALLY went, the
   cells strictly between, in the frame before the press, less the floor
   the mover walks on (ground_ok) and the frame's background.
3. A THING WHOLE ANYWHERE (scene._candidates, placement): a block that
   jumped 21 cells beside a like block was "explained" by near shifts
   onto that block's cells (all taken) and read as vanished, its cells
   grown onto the neighbour and onto the socket frame.  Now any shift
   under which ≥90% of a thing's cells are found is a candidate, and a
   thing whose every near reading falls on taken cells is looked for
   whole on free cells anywhere before it is declared gone.
4. THE READING NAMES ITS MOVER BY FORM before place: among things of
   the kind, those of the reading's size and proportions (m_cells) — the
   small block that wandered near where the tall one stood is not it.
5. A FIT MADE WHILE THE LEVEL GOES ON is a done reading; a done fit whose
   frame stands empty again (its block went to push another) is open
   again, judged by form; after a fit the chase restarts from the
   readings as they stand (no fiddling between dockings).
6. THE ORGAN — plan_by_shove (core_play2): when a thing's own moves
   reach no goal state, every place it can stand alone is a launch, every
   key's strike from it is simulated (slide_shove, factored out of
   push_move and exposed on the successor function), landings ranked by
   nearness to the goal, a pusher (another selectable piece, free ones
   before docked ones) routed behind the launch by its own moves or by
   one strike of its own, the strike predicted with the pusher there,
   the rest routed from the landing (one more strike allowed).  Memoised
   per (thing, state, depth); 60 s budget; up to six searches a level.
   Verified on the real L2 state (S8 refused at the pit): 9 steps in
   0.2 s — S8 to (30,9), S9 up under it, strike, S8 into its socket.
   In the winning run the joint single planner found the strikes itself
   once 1–5 held; the organ stands for the cases it cannot.
Also: _step_is_safe checks the SELECTED thing's step (a pusher on its
way), not the reading's mover.  Reproduced: ka59:1-2 → 80, 56.
Snapshot s40.

## Session 36b (2026-09-21): m0r0 — LIKE MOVERS MEET (L1 won: 41 actions; human 30)

Tushar, guiding step by step: "2 objects which need to be made 1 ... if
you press against a wall, 1 block does not move, another one does ...
hit right (nothing happens) so hit left — you should have kept trying."
The core had learned the mirrored laws exactly (ACTION3: one dot (0,-5),
the other (0,+5)) and then had no reading at all ("pursue: []") and
wandered.  Two organs, both general:
1. LIKE MOVERS MEET (look_again): when the keys move two like things —
   one kind, one form — that answer the keys as separate bodies, the
   reading is that they are to be made one: a "meet" reading whose target
   is the other mover as it stands in each state (dist = the distance
   between the two in the state; neither is a wall to the other).  It
   carries as the template like any winning reading.
2. EACH BODY ANSWERS THE KEY FOR ITSELF (make_successors): a key whose
   law moves several bodies used to be refused whole when any one of them
   met a wall; now a body standing apart from the others (boxes more than
   a cell apart) — or one a press has shown staying while another moved —
   stays while the rest go on.  Bodies that touch are still a chain until
   a press shows otherwise.  The joint search then finds the 16-step way
   in 152 states (the first plan walked into the centre wall once, which
   taught the wall's colour).  A plan step that expects the mover to stay
   is not stopped by the look-before-you-step check.
m0r0 L2 open: 13×13, hazard cells (checkerboard colour 8) that put both
dots back at their start, and the wall's colour differs per level — the
refusal at the centre wall repeated (refused_at / ground_refused not
holding there): to diagnose with frames.  ka59:1-2 re-verified (80, 56).
Snapshot s41.

## Session 36c (2026-09-21): m0r0 L2 — PUT BACK, AND WHAT A CARRIED LAW IS (L2 won: 48 actions; human 111)

Tushar: "this is where I made myself stuck" — the checkered cells send
both dots back to their start.  Seven surgical fixes, each general:
1. PUT BACK (core_play2.step): the controlled things standing at their
   level-start places again after a FAR jump (longer than any step) is a
   reset; what the step ENTERED is learned as a hazard without the loss.
2. HAZARD BY ENTERING (fiddle): death_enter / entered_alive beside
   death_touch / touched_alive — a strip the corridor runs along may be
   stood beside, not on.  The planner's deadly() keeps touch-hazards a
   cell away and enter-hazards only off the body's cells.
3. A REFUSAL IS JUDGED AGAINST THE PLACE BEFORE THIS STEP, not the plan's
   start (a two-step plan's second refusal was never recorded).
4. A KEY LAW BY SELECTION MEANS NOTHING WITHOUT A WAY TO SELECT
   (selection_in_play): two things moved by one law, one of them held by
   a wall, are not a selection (bodies must be disjoint to vote one), and
   a stray key_selected no longer starves the ground learning of intents.
5. A KIND THAT ANSWERED A KEY IN DIFFERENT WAYS carries no single law
   (the mirrored pair); the step length is carried through the keys that
   agreed and the other carried laws take the level's step when a
   confirmed key's step differs (a 5-px board, then a 4-px one).
6. WHAT THE PLAN RELIES ON IS CONFIRMED FIRST: a key whose law here is
   only carried, or never pressed on this level, is pressed once before
   "no way" is concluded — also in the template's early pursuit.
7. NO UNDO VOTE FOR A WALK BACK: a press that moved everything exactly
   back over the last press's steps is the opposite arrow, whatever the
   laws say so far.  Also: like movers of ONE body that stand apart may
   still be read as a meeting (a wall has not parted them yet), and the
   meeting is not "self-aimed"; look_again runs before the template is
   sought so readings that need the mover known are there.
Offline joint BFS on the real map: 20 actions, 1148 joint states; the
core: one walk into the hazard row (learned), then a 29-step plan.
Two things learned the hard way while re-verifying: (a) the probe is
only for keys of which NOTHING is known here (a carried law is verified
by use — probing carried keys moved cn04's pieces and cost it the
level); (b) the early (template) chase keeps the first frame's readings
across dockings — look_again there had never actually run (its def came
later in the function; the NameError was swallowed), and running it
re-read the dockings worse.  ka59:1-2 (80, 56) and cn04:1-2 (18, 67)
re-verified after; m0r0 41/48 reproduced.  Snapshot s42.

## Session 36d (2026-09-21): readings from the replay review (tr87, r11l, re86, cd82)

Tushar's readings are recorded in claude/human-readings.md: tr87 (a
legend of pairs joined by lines answers a query row — "the first one is
the right answer and it has not figured out anything"), r11l ("start with
the extreme ends; select an end, then click far — purple goes there"),
re86 ("look at figure similarity — overlap the pluses, or fit them in
the four dots"), cd82 ("wasted futile efforts").
Done now: the "does the mover answer a click?" probe is made once per
mover thing, not once per reading sharing it (cd82 L1: 91 -> 85; the
paint protocol's 55 trial actions remain the real waste — the picture is
to be READ, see cd82's open item).  re86: the reading Tushar names IS
already made on the first frame ("own-colour constellation" for each
plus and its four dots); what fails is downstream — (a) the rings the
dots sit in were solids on the way to the dots: a small thing drawn round
a mark aimed at is its setting, not a wall (fixed); (b) the two pluses
pass through each other and one bar hides the other's, so the tracker
reads the mover shrunk and the plan diverges; (c) ACTION5 hands the
black centre to the other plus and is not learned as a cycling key; (d)
the arms may hang off the screen, which the successors refuse.  Those
three are the next surgical items for re86.  cn04 18/67, ka59 80/56,
m0r0 41/48 re-verified.

## Session 36e (2026-09-21): tr87 — A TABLE OF PAIRS, BY THE CORE'S OWN PRINCIPLES (L1 won: 32 actions; human 54)

Tushar: "how the heck is tr87 a legend? Implement it. Understand object
rotate/mirror and see what matches — this should be solved with the
existing principle."  Right: no new kind of thing was needed, only the
principles already in the core, composed (agent/core/table.py):
1. WHAT A CONNECTOR JOINS IS A PAIR: a thin short thing of another
   colour whose two ends each touch a tile is a joint; the two glyphs
   it joins belong together.  Six such pairs are a table: form -> form.
2. A FORM IS THE SAME TURNED OR MIRRORED (canon): the least of a form's
   eight drawings names it — the glyphs are drawn at random turns, and
   the same glyph in the table and in the rows still matches.
3. A TILE FRAMES ITS GLYPH: a region of one colour whose box's rim is
   all its colour and whose inside holds one other colour's marks; a
   strip of tiles holds several glyphs (clusters two cells apart); a
   glyph with a hole that reads as a tile framing a scrap is dropped
   because the framing seen more often is the tile.
4. A ROW IS A SEQUENCE; a row of the table's left colour is the
   question, the row of the right colour under it the answer, position
   by position; a small thing of another colour beside one answer glyph
   is the cursor.
5. THE KEYS ARE LEARNED BY ONE PRESS EACH: a press that moves the cursor
   is a move key (its direction from the change), one that changes the
   form under the cursor is a cycle key.  Then each wrong position is
   visited (the shorter way round) and cycled until its form is the one
   its question maps to; the level judges after each edit.
Fires on tr87's first frame only among the 24 real games.  L2 open: a
pair's side may be a CHAIN of tiles (one glyph -> several) — the same
principle extended outward from the connector's end.

## Session 36f (2026-09-21): re86 — THE READING WAS THERE; THE REST WAS EARNED (L1 won: 28 actions; human 26)

Tushar's "look at figure similarity ... fit them in the 4 dots" was
already the core's first-frame reading ("own-colour constellation" for
each plus and its four dots).  What stood between the reading and the
win, each fixed generally and re-verified on cn04, ka59, m0r0, tr87, ft09:
1. A LINE BROKEN BY A DOT IS ONE LINE (entities.read_frame rule D): a
   tiny thing with the same colour above and below it, or left and right,
   is a part of the thing it interrupts — the cross with its black centre
   was four arms and a dot, and the arms had laws of their own.
2. THE RING A MARK SITS IN IS ITS SETTING, NOT A WALL (pursuit solids):
   a small thing drawn round a mark aimed at is not in the way.
3. THE SCREEN IS A WINDOW (push_move): the mover may go partly beyond
   the edge — its arms out of sight, its middle in view — until a press
   refuses it there; what left the screen is kept in the state, and the
   prediction is compared as it would show.
4. ONLY THE SELECTION'S COLOUR CHANGING HANDS IS A HAND-OVER (fiddle):
   once that colour is known, an arm passing over a dot's ring is not a
   selection moving — that junk had made every arrow a "cycle key".
5. A SELECTION HANDED ON BY A KEY completes the other keys' laws
   (complete_selected_keys): the things that have worn the mark are the
   selectable ones, whatever their kinds.
6. A COVER ACHIEVED WHILE THE LEVEL GOES ON is done by its own test and
   the other readings are chased next (a fit stays judged by the thing
   sitting in its frame — nearness alone marked ka59's big block done
   and cost it 38 actions until narrowed to covers).
Also (36e): tr87 L1 by the composed principles.  Snapshot s44.

## Session 36g (2026-09-21): r11l — HUNG BETWEEN ITS HANDLES (L1 won: 7 actions; human 22)

Tushar: "you did the right thing, except you didn't try to also click the
lower one far."  The core's first probe (click the disc, click beside it)
already showed the law a person reads from that two-click experiment:
the clicked-to handle jumps to the click and the disc moves HALF as far.
What was missing was to read that as a law and plan by it:
1. A DOTTED LINE IS ONE LINE (entities rule E): tiny pieces of one colour
   meeting corner to corner are one drawing — only chains where each
   piece meets at most two others (a checkered field, four each, is a
   texture), and only pieces nothing larger has claimed (a line's end
   pixel in a thing's box is that thing's, and must not drag the line and
   what it touches into one).  The handle-to-disc lines were dozens of
   dots; the pluses' arms were parted by their centres (rule D).
2. HUNG BETWEEN ITS HANDLES (fiddle.observe): a thing that moved a
   fraction 1/k of what the clicked-to thing moved, and stood at the mean
   of the k like things' places before and after, sits at their centroid
   — model.centroid_of; carried by form (centroid_kinds); the handles
   become presumed selectable.
3. AIMED BY ITS HANDLES (make_successors): such a thing has no law of its
   own and is in nothing's way; it is put at the centroid after every
   action; the click that moves it is the click that sends the selected
   handle to k·target − Σ(others) — the place itself, else as far that
   way as the screen and the known walls allow, the other handles then
   doing the rest (the search selects them: a 'select' set by the plan is
   trusted; before that, the selected handle is the one whose kind bears
   the click law — a selected handle is drawn otherwise).
4. WHERE A CLICK MAY SEND A THING: the ground under a landing that took
   is walkable; a click that sent nothing where a thing is known to go to
   the click was refused by what lies there — a wall, learned like a
   refused step (the first plan clicked into the red band once).
5. A reading chased before its mover could be moved is chased again once
   the probe has taught how (the early loop retried the disc's reading).
Run: probe (3 clicks), one refused click, then rclick, select the other
handle, rclick — 7 actions.  Re-verified: cn04 18/67, ka59 80/56, m0r0
41/48, tr87 32, ft09 4/7, re86 28.  Snapshot s45.

## Session 36h (2026-09-21): six readings — tn36, lf52, cd82 L2, wa30, vc33 L2 won; bp35 L2 dug out

Tushar's six questions, each answered by a general organ (none names a
colour, a size or a place), the seven won games re-verified after each.

tn36 L1 WON (40 actions; human 32) — "tn36 fits. check why it is not
understood?"  The fits reading was made on the first frame (the block →
the same-colour cup); what was missing:
1. WHAT HAPPENS BETWEEN THE PRESS AND THE REST IS PART OF THE ANSWER
   (agent/core/program.read_journeys): the core read only the last frame
   of each press; the RUN click returned seven frames in which the block
   went down twice and was put back — silent by its last frame.  A press
   that returns several frames is read for a JOURNEY (a thing that
   travelled and stands where it stood): a LAUNCH.  Texture kinds (eight
   or more alike) do not journey.
2. A CLICK THAT RECOLOURS A SMALL MARK IN PLACE, nothing else changing,
   flips a TOGGLE (model.flips colour→colour, model.toggle_cells).
3. THE DISTINCT THINGS FIRST, THE BACKGROUND-LIKE LAST (Fiddler.moves;
   Tushar's lf52 rule): things sorted by how many share their kind, a
   kind strewn eight times or more is texture and goes last, one or two
   of a kind stand for it.  (Hand-chosen: 8, 2.)
4. THE PANEL WRITES THE JOURNEY (program.read_panel/correspond/compose;
   program_protocol): the flipped marks lie in a strip of like slots at
   one pitch (an empty slot counts only between marks); the journey has
   one step per marked slot, in order — a row is a sequence (tr87's
   principle; the legend learned by experiment, not drawn); the journey
   wanted is the reading's displacement (a fit: inside the frame without
   overlap), composed of the steps known, the slots set by clicking the
   differing marks, the launch pressed; a launch that shows another
   journey refines the legend (five tries).

lf52 L1 WON (53; human 32) — "why lf52 is lost? I told you solution":
5. A KIND STREWN MANY TIMES OVER THE BOARD IS THE BOARD: the first-frame
   readings named the floor tiles as movers and the probe spent 40
   presses on them; a mover with eight or more like things is not probed.
6. A MARK THAT APPEARED ON AN OLD THING IS NEW TOO (fiddle+): the ring
   where the selected peg may go was a recolour of an old tile and was
   never clicked; now the mark on it is clicked (not the tile's centre).
7. A LEAP (agent/core/leap.read_leap): two clicks — select, then point:
   the thing under the first click leapt two pitches over its like
   neighbour to the place clicked and the neighbour is gone (read from
   places, not ids: the tracker may swap two like things that pass).
   From that first removal the goal "fewest of the kind left" is planned
   over the lattice of the kind's things and the tiles they stand among
   (leap_protocol; each leap verified by the count).

cd82 L2 WON (70; human 8; L1 89) — "possible by planning if you
understood what happened in L1":
8. THE POUR REACHES THE PART OF THE CANVAS NEAREST THE POURER (paint
   protocol): a geometric prior — the cells nearer the pourer than its
   mirror image through the canvas's centre (a half from a side, a
   corner's triangle from a corner) — checked against the first pour,
   then trusted for every place walked to, each pour verified by use (a
   place where it fails keeps the region seen there).
9. LAYERED POURS ARE PLANNED, NOT GREEDILY FIXED (plan_layered): the
   fewest pours whose layering makes the canvas the picture (what is
   painted first is painted over), among equal counts the nearest places;
   L2's white wedge, black bottom half, purple wedge came from the picture.
   The places are walked first (every key at every place, the inverse of
   the key just pressed known without pressing).  OPEN: L1 still spends
   38 presses on the fiddle and 24 on the walk before its one pour.

wa30 L1 WON (66; human 71) — "why wa30 is still unsolved when we know
push and fit?"  The fit reading was there (crate centres → the pen of
their colour, three abreast); what was missing:
10. A KEY THAT DID NOTHING IN THE OPEN MAY DO SOMETHING IN CONTACT (the
    contact experiment, after the bump block): the body is walked against
    each of the nearest things — beside it along one axis, a diagonal
    neighbour is not faced — the silent keys pressed there, a step away
    read; the thing that came along is CARRIED by the key that bound it
    (model.carry: key, kinds, the body's kind, the thing's dims, its core
    within, the colour it wears while held), and the key lets go again.
11. CARRY EACH THING TO ITS PLACE (carry_protocol): places = as many of
    the thing's boxes as fit in the target's box; each thing is taken
    from the side it can be set down from (the body's own place then must
    be free), bound, planned to its place with the body (its core is what
    is tracked), let go.  Things known by form and place, not id.
    Downstream, each general: a thing's motion is its body's (a blocked
    press is a turn; the sprite steps by four, not three or seven —
    fiddle.observe override into KeyModel); a thing lying within another's
    box rides it whatever their sizes; what is bound to the body is one
    body with it in push_move (moved along, not in its way, its cells not
    walls; a rider not moved twice); a carried thing has no law of its own
    and is not pushable; an overlong step is two things confused, not a
    law (KeyModel: an odd translate over 1.5× the supported step is
    noted, not adopted); the body's step is what it did most often
    (_restore_body_laws before a carry plan); placed things stay solid,
    the target's own parts do not.

vc33 L2 WON (24; human 18) — "vc33 you solve :D":
12. A THING PUMPED DOWN TO NOTHING IS STILL A THING (make_successors): a
    retract that empties a thing removes it from the state and remembers
    its place (`_last`); the extend law regrows it from there; a retract
    of a spent thing refuses the whole key ("a pump with nothing to
    give"); and such a refusal is the pump's emptiness, not the mover's
    place (no refused_at) — the plan refills the middle piston from the
    first and pumps on (7 planned clicks after the fiddle).

bp35 — "why bp35 L3 is lost?": L2 had been lost since a later session
than 31d (both before and after this session's changes, checked against
snapshot s45), and that is what stood in the way:
13. THE WALK PLANNED OVER THE FRAME BEFORE THE LAST PRESS (model._frame
    is the frame before the press; a click that scrolled the screen made
    it another world) and then refused its own first step for ever
    (_step_is_safe without the plan's "an edge is not a wall" rule: the
    body's own grey outline) — the level hung in 50000-state plans.
    Fixed: walks and displays plan over the frame as it is now
    (_frame_now, set at the level's start too), and _step_is_safe has
    the edge rule.
14. IN THE WAY: removals were offered for every thing within two steps
    of the body (the search exploded on subsets of blocks); now only for
    things where the body would go in one or two steps of any key, within
    one step of it on any side, or along its slide.
STATE: bp35 L1 28; L2 dies once on the spikes (as before), then walks
into the pocket under a column of blocks and finds no plan there
(3 states): the removals above are offered but the successor after them
leads nowhere — to diagnose next (WALK_DEBUG from the pocket); L3 not
reached.
BATTERY (after all of it): cn04 18/67, re86 28, r11l 7, ka59 86/56,
m0r0 41/48, tr87 32, ft09 4/7, tn36 41, lf52 53, cd82 66/69, wa30 66,
vc33 26/24 — every earlier win kept; five games newly won (cd82 L2
needed one more fix after the battery: the colour in hand is read from
the holder after a colour click, and from the moving thing's cells when
the holder's identity is lost — a turning basket).
Hand-chosen constants noted: texture = 8 alike; 2 per kind fiddled
first; panel pitch tolerance 0.15; journey needs ≥3 frames; layered
pours ≤4 and 8 s; contact: nearest 3 things, silent keys once each;
carry: 7 rounds; overlong step 1.5×.

## Session 36i (2026-09-21): THE THREE GAUGES, AND THE CORE AS A SUBMISSION AGENT

Tushar: "if we hardcode something, Kaggle changes the games internally
to counter it; make sure hypothesis generation and learning per game can
solve the problem — have we crossed the lower limit for a Kaggle trial?"
Measured, cold, before answering:
- MECHANICS GATE (new core, all 25 first levels): 6/25 PASS; entity
  recall 0.71, rule recall 0.68, goals right 12/25 — while it WINS 19/25
  first levels (session 31d: 7/25 pass, 11 wins).  The play has outrun
  the report's vocabulary: tn36 won and reported as align (truth
  inside), lf52 won with no "count" word, wa30 won and reported reach
  (truth inside), ft09/cn04 won and mis-worded.  Bin 1 again (the
  translation, not the core) — count / program / carry / launch /
  leap are not gate words yet.
- BLIND SET 2 (sx07-12, still sealed; phase-1 record, no key read):
  0/6 first levels with the core (the old naive explorer: 4/6).
- SIM HOLDOUT (sh*, never inspected; counts only): 3/27 first levels
  (the old agent: 18-19/27).
VERDICT: not crossed, and the sealed sets say precisely what was
feared — the organs are stated generally and fire off observed
preconditions, not game ids (that part is right), but the SET of
organs was compiled from these 25 games and the fiddle-and-read loop
does not yet coin a reading on a mechanic outside it.  The core is a
better player of what it was shown and a worse cold-starter than the
naive layer it replaces.
THE SUBMISSION AGENT (tools/build_core_submission.py ->
submission/my_agent_core.py, 764 KB): the core's modules embedded as
sources and loaded as a virtual package (agent.*, tools.core_play2 —
no import stripping, no name collisions); the adapter inverts control
(the core drives a bridge environment from a worker thread; the harness
hands each frame back as the answer to the last action; one game at a
time through a lock, the clock started at its turn; a silent core or
an exception falls back to random legal actions, never a crash).
core_play2.main's level loop is now play_game() (shared by the CLI
and the adapter; persist=True lets the submission try a given-up level
again from where it stands).  Verified through the adapter offline:
tn36 L1 41, cn04 18/67 — identical to the modular runs.  Pre-registered
in SUBMISSIONS.md as v32, [0.12, 0.24], point 0.18, STAGED beside v31,
not submitted (the gate and the two-unaimed rule both stand).
NEXT, in the order the gauges point: (1) the report's words for what
the play already knows (count / program / carry / launch / leap) — the
gate is the cheapest to move; (2) blind set 2 with the core, one game,
to find the first reading the loop fails to coin (M1 with the core is
0); (3) bp35 L2 from the pocket.

## Session 36j (2026-09-22): THE POST-MORTEM — "why the crude agent scored and the core does not"

Tushar: "so we made the core worse? .30 also read .21; analyze and
publish what made the crude one better and the current one bad and the
direction; do not rewrite."  Published as the project doc "Why the crude
agent scored and the core does not" (a Claude doc; also in claude/).
In one line each: every Kaggle score of the old line sits on one floor
(0.21; V3's 0.30 now reads 0.21 too); the crude loop never stops and
never thinks long — probe, unique thing, frontier, class-scored clicks,
death ledger, random known motion, for 4000 actions — and that earns the
floor because RHAE charges nothing on unfinished levels; the core reads
or leaves (no_plan after 16–74 actions on the sealed set) and has no
floor; the drift was five decisions (rewrite as a separate harness,
never layered; the gate measured description, not cold play; the 25f
warning noted, not acted on; ten sessions of organs verified only on
the won games).  Direction: layer the core over the crude loop (a
dispatcher, not an organ); no_plan is a hand-off, never an end; cheap
actions before long plans; the three gauges become the battery with the
floor as the bar (4/6, 18/27); then M1 one sealed game at a time;
submit only the joined agent once it beats the floor.  No code changed.
Tushar's review of the crude-loop list (2026-09-23) folded into the
doc's direction as three more rules: register what causes what before
planning (the core does — laws per key/kind, refused_at, hazards by
kind, carried across levels; what it lacks is the floor's per-action
ledger of what did and did nothing); the goal candidate is the unique
thing whatever it is made of (a pixel constellation, a hollow, a
socket — the relation language's things, not connected components);
when exhausted, remember what failed and never repeat it (the floor's
escalating death tail, two-witness class condemnation, prefix bans;
the core's retry carries laws but not the failed plan — add a trial
memory per level so each retry differs by at least one move).  Still
no code changed.

## Session 36k (2026-09-25): THE FLOOR UNDER THE CORE — the dispatcher

Tushar: "implement as discussed on current gaps."  Done in one place,
`play_game` (tools/core_play2.py), the loop that drives both the
command line and the submission; no organ of the core changed.

WHAT WAS FOUND FIRST (engine, verified with a script): a RESET after a
death is a LEVEL reset, but a second RESET with no action in between is
a FULL reset to level 1 with score 0 (`handle_reset`: action count 0 →
full).  The old `play_game` sent exactly that second RESET
(`env.reset()` after the death's own RESET inside `step`) — that is the
whole "one more attempt from the first level, replaying the won
levels".  Self-inflicted; gone.  The floor's driver and the dispatcher
keep `since_reset` and never send the second RESET.

THE DISPATCHER (`play_game`): the core plays a level; if it wins, its
export carries to the next.  If it leaves the level (no_plan, or a
death), `Floor` — the crude loop, `MapperAgent` v11, one instance per
game — plays from where the level stands, for a turn of FLOOR_TURN=150
actions that doubles per turn on the same level (max 1200); the floor
handles its own deaths (RESET, on_game_over) and keeps its ledgers
(death tail, prefix bans, click classes) across turns — re-sighted with
`observe(RESET)` when it gets the level back, `on_level_up` on a new
level.  The core is given the level again only from a STATE it has not
tried (frame hash), at most CORE_TRIES=3 times per level; after a
death the core looks again at once only if the death TAUGHT a hazard
(`_hazards_of(export)` grew) — a death that taught nothing (the clock,
a wall) would be the same trial again (sp80: the core died at 30 twice
before this rule; now once, then the floor wins in 11).  A level is
never given up while the game runs: the loop ends at WIN, the action
cap (`max_actions`, 4000 in the harness) or the wall clock
(`seconds`).  When the floor wins a level, the core's laws from that
level still carry (how="floor").  In the harness the core lock is held
per core turn only (`lock=`, CORE_WAIT_S=20): while one game's core
thinks, the other games play their floors.  CLI: `--no-floor`
(diagnosis, the core alone), `--max-actions`, `--seconds`.
Hand-chosen constants: FLOOR_TURN 150, FLOOR_TURN_MAX 1200, CORE_TRIES
3, CORE_WAIT_S 20.

MEASURED (400-action cap, the floor's own measurement conditions;
counts only, sealed): blind set 2 L1 4/6 (sx07 08 09 10; core alone
0/6; floor bar 4/6) + sx10 L2; sim holdout L1 21/27 (core alone 3/27;
floor bar 18–19/27) — 5 by the core (shf2, shj0, shu0, shu1, shu2; two
after a death that taught), 16 by the floor.  Visible: tn36 L1 41
(core, unchanged), sp80 L1 won (30 core + 11 floor; core alone lost).
Bundle rebuilt with the floor's modules embedded
(tools/build_core_submission.py → submission/my_agent_core.py, ~1.1 MB);
adapter check through the harness loop: sp80 L1 won.

THE GAP THAT REMAINS (item 6 of the post-mortem): the core's own memory
of a failed plan.  The dispatcher forbids a repeated trial from the
same state and a retry a death taught nothing, and the floor's ledgers
persist — but inside a core attempt the pursuit can still search the
same way it did last time from a new state.  Earn it on a level where
that is the loss.

## Session 36l (2026-09-25): THE WINS MUST TEACH SOMETHING — the core watches the floor, reads its wins, confirms by replay

Tushar: "go with this next thing ... the wins must teach something.  You
can even internally play the game again to confirm if your learning was
correct, to confirm the robustness of the hypothesis derived."

THE CORE WATCHES THE FLOOR (`watch_floor`, core_play2): every press the
floor makes is fed to the core's tracker and model exactly as its own
presses are (`model.observe`, `note_alive`/`note_death`), so the laws
of the keys, what clicks do and the hazards learned during the floor's
turn are the core's when it gets the level back and carry to the next
level.  A journal of the floor's presses (key, click, events, things
before/after) is kept for the reading of the win.

THE WIN, READ (`_reading_won`): when the floor wins, the level's own
readings are tested against the state before the winning press — a
thing of the reading's mover colour that moved in the level (or wears
the mover's colour), at / in / over / against the reading's target; the
thing standing on the target's own cells unmoved is the target, not a
mover.  The first such reading is the won reading and carries exactly
as the core's own wins do (`won` template pursued first on the next
level; relation weights raised).

THE WIN, COINED (`coin_from_win`): when no reading matches, the win is
read from the trajectory: the mover is what the keys move (found by its
colour when the tracker lost it); the winning press's law says where it
went — into / over / against which thing (an ARRIVAL); things of one
colour and form that changed or vanished as the mover reached them, the
winning press reaching the last (a COLLECT); a winning click names the
thing clicked (a CLICK).  The coined class (kind, mover colour, target
colour, target dims, relation) rides the carry (`coined`) and on later
levels becomes readings on that level's things ("won before: collect
C5[..]" / "won before: reach C8[..]") pursued first through the template;
a coined click puts things of that colour first in the fiddle
(`Fiddler.moves(first=)`).

IDENTITY BY COLOUR AND FORM (`_merged_mover`, `SceneTracker.rename`):
found on sx09 from the play record (no key read): a mover entering a
thing of its own size was read by the tracker as "that thing
recoloured, the mover vanished" — so no move, no passing-through, no
portal was ever learned from it.  Now a mover that vanished on a press
while exactly one thing took its colour and its form (at the law's
place, or anywhere — a portal) IS the mover; the tracker renames it and
the event reads as its move.  Applied in the core's own `step()` and in
the watching.  Regression set unchanged (tn36 41, lf52 53, wa30 66,
cd82 66/69, vc33 26/24, bp35 28, su15 14, m0r0 41).

CONFIRM BY REPLAY (`tools/confirm_lesson.py`): run 1 the joined agent to
level L (the level's export = the lesson: the won/coined reading, the
laws, the hazards); run 2 a fresh game, the floor alone to the level,
then THE CORE ALONE seeded with the lesson.  Confirmed = the core wins
by itself from what the floor's win taught.

MEASURED (sealed, counts only; 400-action cap): blind set 2 first
levels 4/6 as before, now + sx09 L2 and sx10 L2; every floor win now
reads: sx07 "fits frame C9", sx08 "fits C8", sx09 L1 and L2 both "mover
12 fits C8" (the same reading explaining both levels' wins — the
hypothesis is consistent across levels, though the core alone still
cannot plan it: the level needs portal exploration), sx10 L1 and L2
coined "click C5 (3x3)".  Sim holdout first levels 20/27 (21 before:
shg0 fell to the floor's variance; the floor bar is 18–19).  Of the
floor's 15 sh wins: 8 read as the core's own readings (6 collect, 1
align, 1 cover), 2 coined (arrive), 5 unread — the M1 list.  CONFIRMED
BY REPLAY: shf1 (collect C8: the core alone, seeded with the lesson,
wins in 31 by its collect protocol — alone without it, it died at 40)
and shw0 (collect C8: wins in 34).  Not confirmed: shj1 (collect C2:
dies at 5 — the lesson carried no hazard), shv2 (fits C8: no_plan),
sx09 (fits C8: no_plan, portals), sb26 (sequence: unread).

Other fixes on the way: a paint_protocol KeyError (region_test) guarded;
a core fault inside `play_level` no longer kills the game — the
dispatcher reports "crashed" and the floor plays on (`_Bridge.last`
gives the frame it stands at).  Hand-chosen constants: none new.
Bundle rebuilt (v33, ~1.1 MB), adapter check sp80 L1 won.

WHAT THE NUMBERS SAY: the floor's wins now teach in the core's own
words on 10 of 20 sealed wins, and on 2 of the 6 tried the lesson alone
lets the core win the level — the first M1 > 0 evidence there is.  What
they do not yet do: fix the core's planning where the reading is right
but the level needs exploration (portals: sx09; a hazard the win did
not meet: shj1).

## Session 36m (2026-09-25, later): THE SLIDE LAW, THE CLEAN SECOND TURN — from the confirm battery

Tushar: "go ahead."  The confirm-by-replay battery was run over every
floor win (sealed sets: 21 cases; the inspectable tuning sims st*: 33
games).  Confirmed on the sealed sets: shf0 (reach the odd thing →
display), shf1 and shw0 (collect); on the tuning sims, one COINED
lesson confirmed (st22: coined collect C8 → the core alone wins in 18)
and every core-won level re-won from its own lesson.  The failures
sorted: "the reading is right, the plan is none" (9 sealed, 11 tuning),
deaths (5), unread wins (3).  On the tuning sims the plan-is-none class
was one mechanic: SLIDE movement (a press moves the thing until
something stops it) — no law for it in the core.

THE SLIDE LAW (planner.KeyModel): two moves of one key on one thing the
same way but not the same distance, both whole multiples of a step that
is at least 2 and one of them several steps, make the law the STEP
(gcd of the moves and the thing's extent along the axis) with
`slides[(key, sid)]` set; a move of another whole length is then
support, not contradiction.  The plan (make_successors) repeats the
step while push_move allows it — the thing goes where it stops.  A
first version took a 3-cell blocked step on wa30 for a slide (gcd 1):
the whole-steps rule fixed it (wa30 back at 66).  Carried across levels
by kind (`slides` in export/seed).  THE SLIDE TEACHES AT BOTH ENDS
(fiddle.observe): what a slide crossed is ground (ground_ok); the cells
one step on from where it came to rest refused it — a thing there is a
wall of its kind, else the ground colour is refused (ground_refused) —
so the plan knows the walls a slide stops against without a refusal at
every place.  st32 (identity/slide/reach/none), never planned before:
the core alone WON in 21 (plan 9/9).  Tuning sims, the core alone:
24/33 first levels (21 before; st32, st81, stc0, stc2 new).  Regression
(21 public levels): all unchanged.

THE CLEAN SECOND TURN (play_game): counts showed the floor winning
shh0 in 9 from the level's start and losing 400+ after the core's 15
presses — not the state (15 random presses left it winnable in 9) but
the floor's own first reading of a level seen mid-way, which its
persistence keeps through its resets.  Now the floor's second turn on
a level begins with one RESET (a level reset: the start again, nothing
banked lost) and a fresh floor mind (`Floor.play(fresh=True)`); the
first turn still plays from where the core left it.  shh0 9, shg0 91
(their from-scratch numbers) after the rule.

MEASURED (sealed, counts only, 400-action cap): sim holdout 21/27
(the core alone 7: shf2, shh2, shi1, shj0, shu0, shu1, shu2 — 5 before);
blind set 2 first levels 3/6 at 400 (sx08, sx09, sx11 — sx11 new; sx07
and sx10 lost at the cap: sx07 WON at a 1200 cap — the floor's 304 from
a fresh start; sx10's earlier win at 87 was a state the old core turn
left and the new one does not) + sx09 L2.  The cap is the floor's own
measurement condition; the harness allows 4000.

Hand-chosen constants: the slide step ≥ 2 and one move ≥ 2 steps (the
whole-steps rule).  Bundle rebuilt; adapter check sp80 L1 won.

## Session 36n (2026-09-25, evening): THE DEATHS CLASS — six general faults, from the tuning sims

Tushar: "okay sir please proceed."  The confirm battery's second class:
a death on the first look, no win after.  Diagnosed on the tuning sims
(st20, st70; nothing sealed inspected).  Each fault general:

1. BLAME REACHES NO FARTHER THAN THE MOVE (fiddle.note_death): the ray
   along the key's direction ran 64 cells, so a walker's death blamed the
   collectible far down the row (st20: [2, 8] — the goal condemned, no
   reading left).  A step's ray is the step's length; only a slide's
   goes on, and for a slide the LANDING is found (the first blocking
   colour or solid thing) — what the landing stands on is ENTERED, what
   lies beyond is touched.  `note_death(frame=)`.
2. A LOST LEVEL'S FRAME WITH THE MOVER GONE TEACHES NO LAW (step,
   watch_floor): it read as the key doing nothing, and a carried law
   "refused twice" was dropped.  With the mover still shown, the press
   teaches as any other (the first press of the fatal key is often the
   only sight of its law).
3. A KEY NEVER PRESSED IS ASKED ONCE on a carried level (the confirm
   phase stopped at the first key that moved; ACTION4 never pressed
   before the death was never pressed after — a hole in the repertoire).
4. LAWS CARRIED IN AND UNCONTRADICTED CARRY ON (fiddle.export): an
   attempt that did not press a key no longer unlearns it (st70: the
   keys forgotten between two looks).
5. ONCE MORE FROM THE START, KNOWING (play_game): after a no_plan on a
   level whose start was spent by blind fiddling, one level RESET and a
   look with everything learned (st70: the goal one press left of the
   mover's start).
6. PLAN BEFORE THE FIRST PRESS on a carried level: with the laws
   carried and the mover presumed as the last win's mover colour, the
   readings are planned from the start; the plan's presses confirm the
   keys by use, and no blind confirming press follows them (cn04 L2: a
   docking undone by one — caught by the regression, fixed: 65 vs the
   baseline's 67).  And BLIND PRESSES AVOID KNOWN HAZARDS
   (`_press_deadly`, the planner's own rule): the fiddle's repertoire
   and the confirming presses skip a key whose known step lands on or
   beside a kind that has killed (st20: the second look dying as the
   first).

RESULTS.  st20 (died at 3 → WON 38, collect), st70 (died at 19 → WON
in 1 from the start) by the core alone.  Tuning sims, the core alone:
28/33 (24 this morning, 21 at the day's start; st22, st60, st72, st80,
st82 left).  Two-level regression on 19 public games against the
start-of-session snapshot (s47): every first level as before; second
levels won as before (cd82, vc33, m0r0, ls20, ka59, cn04, ft09) — cd82
L2 now sometimes needs the restart look (47 lost + 57 won); the second
levels lost were lost in the baseline too (tn36, lf52, bp35, su15,
tu93, s5i5, r11l, ar25, lp85, re86, tr87 — ar25 L2 was won once in an
intermediate run: variance).  SEALED (400 cap, counts only): sim
holdout 23/27, ELEVEN by the core alone (7 this afternoon, 3 at the
session's start); blind set 2 first levels 3/6 (sx08 floor, sx09 floor,
sx11 CORE in 5 actions, plan 5/5) + sx09 L2 (floor) + sx11 L2 (CORE,
6 actions) — the first blind game the core solves by itself, two
levels.  Bundle rebuilt (v33), adapter check sp80 L1 won.

## Session 36o (2026-09-26): THE HARNESS FIT — the timing was never tuned for Kaggle

Tushar: "are you sure that the timing threshold per level is tuned
right in current implementation?" — No.  V12 (v33) scored 0.24, V10's
number; and the leaderboard's leader stands at 19.45 on the same
column: the official metric is percent, our 0.21–0.24 is a quarter of
one percent (one or two hidden first levels), and the ledger's earlier
readings of that number as a fraction were wrong (corrected in
SUBMISSIONS.md under V12).  Others score below 0.21, so the band is
not a hard floor either — it is where every agent lands whose real
play never reaches the hidden games.  WHY OURS DID NOT: in the harness
all 110 games start at once as threads (the swarm has no worker limit,
no per-action timeout); the bundle held ONE core lock for all of them,
a game that could not get it within 20 s played the floor, a core turn
could hold the lock 600 s, and the game clock was 100 minutes from a
common start — so on Kaggle the core got a few turns in a few games and
the score was the floor's.  Every constant had been chosen for local
diagnosis, one game at a time; the sealed gauges were measured under
conditions the harness never provides.

THE FIT: (1) the core's per-level state on its functions
(`play_level.last`, `make_successors.whys`, `plan_general.last`,
`play_game.carries`) is now PER THREAD (agent/core/perthread.py,
`PerThread`), so games think in the core concurrently; (2) the adapter
holds a BoundedSemaphore of min(8, cpu_count) core seats, not a lock;
(3) the game clock derives from one run budget, `ARC_RUN_BUDGET_S`
(default 6 h — hand-chosen, half of a 12-hour notebook; raise once the
Code Requirements page's limit is read), every game's deadline the same;
(4) a core turn thinks for TURN_FRACTION (0.25) of the game's remaining
clock, at most LEVEL_SECONDS (600), and a plan search for a quarter of
the turn (≤ 90 s); (5) the floor's turns cap at 600 actions (a level
won after 600 floor actions scores nothing under the quadratic rule;
the turn is for reaching the next level) and the floor may spend at
most FLOOR_SHARE (0.5) of the level's remaining actions while the core
has untried states — then the game waits its turn at the core instead
of burning actions; (6) ACTION_WAIT_S 1800.

THE DRESS REHEARSAL (tools/rehearsal.py — the thing never run before):
the bundle played as the harness plays it, eight games at once through
MyAgent.choose_action, two seats, a 25-minute budget: every game got
core turns (1–11); levels: tn36 1, sp80 1, sb26 1, st70 2, st32 2
(the game won in 23 actions), sx09 2, shf1 2, vc33 2.  On Kaggle (4
CPUs, 110 games, 6 h): about 13 minutes of core time per game — a few
turns on the first levels, against effectively none before.
Regression (core alone): tn36 41.  Bundle rebuilt (438 KB).

## Session 36o addendum (2026-09-26): what the Duck-harness forks say (two public notebooks, 3.95 and 3.92)

Tushar shared two forks of Tufa Labs' "Duck" harness (an LLM agent — a
Qwen 3.8 NVFP4 model served by vLLM on an RTX Pro 6000 — 28 games at
once, 2.2 h per game, no action cap).  Worth knowing:
- THE NOTEBOOK LIMIT IS 9 HOURS: the harness gates on
  `target.max_runtime_s == 32400` and a fork's comment says the Save &
  Run budget is 8 h 50 m.  v34's run budget is now 7.5 h (was a 6-h guess).
- `ONLY_RESET_LEVELS=true` is pinned before the client is built ("RESET
  keeps the level"): the engine's handle_reset honours it (no full reset
  ever), and arc_agi's API marks the scorecard accordingly.  Set in the
  bundle at import.
- THE PUBLIC BASELINES per level are the metadata's `baseline_actions`;
  the harness prints agent/baseline per level.  Our core alone on the
  public 25 by the official formula (level-index weights, 115 cap):
  ≈ 4.1 % mean; the Duck's own public-25 run: 7.20 % (198 000 s of
  game wall-clock).  Hidden: Duck 3.95, ours 0.24 — their transfer
  ≈ 55 %, ours ≈ 6 %.  The harness fit is the lever that difference names.
- Where our public score leaks: first levels won far over baseline
  (vc33 26 vs 7, ka59 86 vs 28, s5i5 47 vs 20, lf52 53 vs 32 — the
  fiddle's actions count), and the level-index weighting (a first level
  of a 7-level game is 1/28 of the game): second and third levels are
  where the score is (ft09 5/6 levels gave the Duck 71 % on that game).
- The Duck won 0 of 25 public games outright; its strength is reaching
  levels 2–5 on the games it reads (ft09, vc33, lp85, tu93).

## Session 36p (2026-09-26): WHY THE OTHER GAMES ARE NOT WON — the diagnosis, no code

Tushar: "our approach is how a human creates hypothesis through
experiments and feedback ... if perception is right, the hypothesis of
what makes sense should be right too. Let's figure out why our other
games could not be won — not through curve fitting."  Every public
level the core alone loses was replayed with its trail and read against
study/truth (engine-derived human accounts; never read by the agent):
the four first levels (dc22, sc25, sk48, g50t) and the eleven second
levels of games whose first level it wins.  Published as the doc "Why
the other games are not won" (claude.ai/code/artifact/c86d290b-...).

THE FINDING: perception is right on nearly every lost level and the
reading right on about half; the break is between experiment and law
(an effect at a distance filed as "silent click, new thing" — dc22,
g50t, sc25), and on second levels the METHOD does not carry (lf52 leap,
tn36 program, tr87 table), the mover is not re-established (re86,
ar25), a probe is repeated from the same state (lp85, sc25, re86), or
the fiddle's own caps hide the controls (tn36's 24 toggles = texture).
Nine principles ranked by levels touched; build order: the exhaustion
rule inside a level; ask every kind once and a boxed array is a panel;
the method carries; the mover re-established; then the SWITCH LAW (an
action toggles a set of things between seen configurations, planned
over) — the one that opens a family.  Beyond the vocabulary: merge to
the next of a legend (su15), sequence to a reference (sk48, tr87),
chains (s5i5), a replaying copy (g50t).  Awaiting Tushar's pick.

## Session 36q (2026-09-26): THE RERUN, REHEARSED — the hidden set never saw the core

Tushar, with V14 = 0.24 (the harness-fit bundle, unchanged from V10/V12):
"so do you have a plan on what needs to be fixed not curve fitted? Also,
board reading and perception must be right."  Before a plan, the question
the five identical scores asked: what does the hidden run actually
measure?  Two tools, no agent change:

- tools/harness_run.py — the bundle through the real
  `agents.agent.Agent.main()` of ARC-AGI-3-Agents, offline (the package
  `__init__` imports langgraph; a bare `agents` package is registered so
  `agents.agent` loads alone).  Result: no fault; tn36 L1 41, sp80 L1 by
  the floor — the loop itself is not the leak.
- tools/gateway_server.py + tools/gateway_run.py — the arc_agi REST
  server in COMPETITION mode as a separate process (what the Kaggle
  gateway is: `_competition_games()` in the Duck notebook builds the
  Arcade over `ARC_BASE_URL`), and the bundle over HTTP through
  `Agent.main()` in the Swarm's shape (every env made first, one thread
  per game, all at once), the server's own scorecard read at the end.
  Facts learned from the server: RESET on a level's first action is
  answered with the observation (no full reset is possible in competition
  mode; `ONLY_RESET_LEVELS` is the GATEWAY's flag, ours does nothing
  there); every request has a 10 s timeout and a None frame raises in
  `main()` — one gateway stall ends that game's thread; the scorecard
  auto-closes after 15 idle minutes (some game must act); the card score
  is the mean over all environments of the level-index-weighted level
  scores (formula confirmed on our own numbers: tn36 L1 41 actions,
  baseline 32 → 60.9 × 1/28 = 2.18).

THE FINDING (104 games at once, 1500 s budget, the v34 bundle): the
public games' mean score 0.47 with 14 levels — against ≈ 4.1 one game at
a time — and the core played NONE of them: the eight core seats went to
whichever games' threads reached them first (the cheap sims: 107 levels,
mean 44), every public game "played its floor meanwhile" (FLOOR_SHARE,
CORE_WAIT_S = 20 s) and the floor took the first levels at 500–1300
actions each (lp85 L1 in 914 against 17: score 0).  The rerun starts
~110 games at once; Python threads share one interpreter lock.  That is
0.24: the floor alone, for five bundles.  Not readings, not timing per
level — the seating.

THE FIX (v35, the adapter only — tools/build_core_submission.py):
- ONE GAME AT A TIME PER SEAT.  A seat is a whole game's turn at the
  CPU, held from its first action to its last, the core AND its floor;
  the others wait without acting (the harness has no per-action
  timeout; waiting for a seat is not "silence" for ACTION_WAIT_S).
  `_SEATS` = 2 (ARC_SEATS; hand-chosen: one thinks while the other
  waits on the gateway; the lock makes more seats no faster — load
  average 1.0 on two cores).
- FAIR SLICES.  A seated game's clock = (deadline − now) × seats /
  games not yet done; games that finish early leave their time to the
  rest.  Kaggle: 27000 × 2 / 110 ≈ 490 s a game.
- NO RANDOM AFTER THE TURN.  A game whose worker has finished waits out
  the run and gives its one RESET at the deadline (the random fallback
  only loaded the gateway the seated games talk to); random remains the
  fallback for a fault.
- FLOOR_SHARE / CORE_WAIT_S / `lock=` in play_game are unused by the
  bundle now (the command line never used them).

REHEARSED (25 public games at once over the gateway, 3000 s → 240 s
slices, shorter than Kaggle's): public mean 6.66, 30 levels — ft09 all
six levels by the legend organ (100), m0r0 13.5, ls20 10.4, cn04/r11l/
tr87 4.76, sp80 4.1, vc33 4.3, lp85 3.5, cd82 3.4; 0 faults.  Against
0.47 for the same games under the old seating.  Second levels not won
one game at a time are still not won (the slice is a quarter of the old
clock: re86 L1 died at 117 where alone it plans in 28 — the plan search
ran out).

NEXT HARNESS STEP (not in v35): seats as PROCESSES (fork at seating,
the bridge over a pipe, actions sent as (id, data)) so the seats use the
cores the lock hides — ×2 here, ×4 on Kaggle — rehearsed the same way
before it ships.

ALSO: tools/perception_report.py — THE PERCEPTION GAUGE: the tracker's
objects on a level's first frame as a numbered picture and a table
(/tmp/perception/<game>.png, .txt), for Tushar to read against his own
eye; every disagreement is a diagnosis (cn04's first look: the two tabs
are separate things from their T until they move with it — a person sees
one piece).  The plan doc: "The plan — what is fixed, not fitted".

## Session 36q addendum (2026-09-27): the second look at the timing

Tushar: "double check the timing and everything. It is not like if I
get infinite trial you are going to really solve all the levels."  Both
true.  Checked against the notebooks and the server code:
- The rerun's budget is 32 400 s (the Duck notebook asserts it); the
  Duck ran 28 games at once at 7 920 s each ≈ 8.8 h.  Ours: 7.5 h from
  the module's import, 1.5 h of margin for the notebook's setup and the
  swarm's close.  Kaggle's seats: 27 000 × 2 / 110 ≈ 490 s of wall a
  game, half a core each under the lock (≈ 245 s of thinking; the
  rehearsal's slices were 240 s of wall).
- Three holes found and closed in the adapter: (1) after the last game
  the threads sat silent until the deadline — the gateway closes a
  scorecard idle for 15 min, so the final RESETs and the swarm's close
  would have hit a closed card; now `is_done` ends the run when every
  game is done.  (2) A gateway request past its 10 s timeout returns
  None, the harness's loop raises, and the game's worker would have
  held its seat for ever; now `main()` is wrapped to release the seat,
  and `do_action_request` asks three times before giving up.  (3) The
  bridge is closed for good on stop, so a worker mid-thought cannot
  re-enter.  Verified over the gateway (ft09 all six, vc33 two, wa30
  one; the run ended at 360 s of a 400 s budget once every game was
  done).
- What time buys, measured: one game at a time with 600–700 s a level,
  21 first levels and 7 second levels; 240 s slices, 19 first levels
  (ka59 and re86 need the longer plan search).  Beyond that, time buys
  nothing: the ceiling is the vocabulary, which is sections 3–5 of the
  plan.  The hidden games' mechanics are new, so v35's score is the
  first measurement of transfer, not of time.

## Session 36r (2026-09-27): V15 = 0.00 — THE SEAT BECOMES A TURN, NOT A GAME

V15 (v35, whole-game seats) scored 0.00 and its run ended in hours, not
7.5.  The confirmation I gave was against my rehearsal path; the rerun
differs from it.  Tushar sent the notebook's page and logs: the commit
log only installs the wheels and writes /tmp/my_agent.py (the games are
played in Kaggle's rerun, whose log we never see).  The page shows the
rerun exactly: the framework copied from the competition data, our
bundle dropped in as agents/templates/my_agent.py, agents/__init__.py
rewritten to register it, a .env with OPERATION_MODE=online,
ARC_BASE_URL=http://gateway:8001/ (trailing slash) and a fixed key, then
`python main.py --agent myagent` — the Swarm over every game the
gateway lists; "the gateway records every action and emits
submission.parquet".  tools/kaggle_run.sh now reproduces that path
verbatim against tools/gateway_server.py.  v35 passes it locally (tn36
L1, ft09 all six), so the break is in what the gateway sidecar does that
the local server does not.

THE ONE THING v35 DID THAT NO EARLIER BUNDLE DID: leave ~108 games idle
for hours after their opening RESET (a whole-game seat; two games at a
time).  v34's games all acted from the first minute (the floor), and
scored 0.24.  I cannot read the sidecar, so the idle wait is the
suspect by elimination, and the design that removes it is better on its
own terms anyway:

THE SEAT IS A TURN.  Every game plays at once, one thinking at a time:
a game takes the one seat to think out its next action, gives it up
while the gateway answers, and takes it again in turn (a fair queue —
`_FairSeat`, first asked first served, a holder past HOLD_MAX = 600 s
abandoned to the next).  Every game acts from the first minute to the
end.  The core's clocks — a level's turn, a plan search, the game's
budget — run on a SEAT CLOCK that ticks only while the thread holds
the seat (`time` replaced in tools.core_play2 and agent.core.planner by
a proxy; the bridge accounts the waits), so a game's budget is its share
of thinking time: 27 000 s / 110 ≈ 245 s on Kaggle.  The watchdog counts
silence only while the seat is held (ACTION_WAIT_S 660).  Kept from the
second look: the run ends when every game is done, a missing frame is
asked for three times, a dying harness loop closes its bridge.

REHEARSED ON THE NOTEBOOK'S PATH (tools/kaggle_run.sh, 25 public games,
3000 s → 120 s of seat time a game): public mean 6.68, 32 levels, 0
errors — ka59 and re86 first levels back (lost under the 240 s wall
slices), tu93 to level 3; ft09 all six.  All 25 games acting throughout;
the run ended 4 min after its budget.
Then 104 games at once on the same path (1200 s → 11 s of seat time a
game): all 104 acted, 89 levels, public mean 3.51, 0 errors, the run
ended on its budget.  Delivered as v36 (ledger: [0.3, 3.5], point 1.2).

## Session 36r addendum (2026-09-28): V16 "Kaggle Error"

v36 came back "Kaggle Error" (not "Notebook Threw Exception", not
"Timeout") — Kaggle's label for a failure on its side of the rerun, and
the one label that carries no information about the agent.  Checked on
ours: the notebook's install cell is byte-identical to V15's (it is not
the cause); memory with every game live at once — 104 games at 23 s of
seat time each peak at 1.86 GB here, one game over 600 s and 2 000
actions at 270 MB, the seat serialises plan searches so their memory
does not multiply; an estimate for 110 games over the full run is under
10 GB against Kaggle's 30.  Two hardenings that change no play: the
harness's per-game frame list is trimmed to its last four (it keeps
every frame of every game otherwise), and the last RESETs, which all
games reach together, are spread over 90 s instead of a burst at the
gateway.  Delivered as v36b; the ledger row stands.  If it errors again
the same way, the fault is ours and the next bundle differs from V14 by
one line.

## Session 36s (2026-09-29): V18 failed at 42 minutes — the silence of one seat

A 2 h run of the round-robin seating failed at 42 min exactly as the
7.5 h runs had: the fault is in the mechanism, not the length.  What
the mechanism does that no earlier bundle did: with ONE seat and no
limit on a hold, the entire run falls silent while one thinker thinks
— measured on the public games over the notebook's path, gaps of 91 s
and 167 s between any two actions within ten minutes.  On 110 unknown
games a single unbounded think (a frame of hundreds of objects, a
reading with no time cap) stalls every game's actions for minutes, and
whatever the gateway sidecar does with minutes of silence, no bundle
that scored ever showed it any.  v36d: HOLD_MAX 45 s (the next in line
thinks alongside; the long thinker finishes on its own time — under
the interpreter lock they share the core), ACTION_WAIT_S 240 s (a game
whose thinker is stuck plays on without it), the worker released on
fallback, the 7.5 h budget back.  25 public games, 60 s of seat time
each: 4.69 / 22 levels / 0 errors, largest silence 91 s.  The deeper
fix, when the harness is settled: a think budget inside the readers
and the successor generation, so no single action's thinking is
unbounded — a general organ, not a harness patch.

## Session 36s addendum (2026-09-29): V19 = 0.20 — the measurement

The seating held (Succeeded, full run) and the core, playing every
hidden game with its fair share of thinking, scored inside the floor's
band.  Six harness bundles to reach the first honest number, and the
number says the vocabulary does not transfer.  What remains true: the
harness is now known and reproducible (tools/kaggle_run.sh); the floor
alone is 0.21–0.24; the leader is 45.33 with an LLM on a GPU.  The
next decisions are Tushar's: the invariance gauge (does the core even
survive recolouring and mirroring of the public games?) is the
cheapest test of whether anything here generalises; the leaders'
notebooks are public tomorrow.

## Session 36t (2026-10-02): THE INVARIANCE GAUGE

Tushar's criterion made a tool: tools/invariance.py (`--transform`
on core_play2) — the same game recoloured, mirrored, turned.  Results
and the four general fixes it produced on the first day are in
invariance-gauge.md (project doc).  In one line: colour 18/18 (after
one fix), mirror 14/18, quarter turn 10/18.  The core is colour-blind
and orientation-bound; the orientation-bound organs are the next
diagnoses, one at a time, each confirmed on the plain game and all four
transforms.

## Session 36u (2026-10-03): THE ATTEMPT LEDGER — the agent's own failure report

Tushar: "we need actionable failed report. Why it failed and how to not
get into the same situation again."  agent/core/ledger.py: a plan is a
prediction; the executor already simulated each step before pressing
and compared the mover's predicted place with the observed one, but
only to replan.  Now every such divergence is RECORDED (step, action,
thing, expected, observed, what stood in the way, the law that made
the prediction, the planner's reason), and at the end of a lost
attempt the core writes its report — "step 1 ACTION2: S30 (kind 3)
expected at (40, 13), stayed; in the way: S13 (kind 6)" — and acts on
it: the law the first divergence contradicts (passable/ground/flat for
the kind in the way; else the key's law for the mover's kind; a
relative-click law for a click that failed to send) is DOUBTED — left
out of the laws the next attempt is seeded with, to be re-learned by
observation — so the next attempt differs from this one in exactly
the belief that failed.  Where no law by kind is named, the refusal
stays by place (as before).  A no-plan ending reports the readings
tried, the keys with laws, and the kinds never asked.

Public regression unchanged (21 first levels; cd82, ft09, ls20, m0r0,
vc33 second levels at 300 s); 36 reports written, compiled for review
in failure-reports.md (project doc).  Not yet: the DIRECTED re-test —
a probe aimed at the contradicted law before replanning — and the
track record (confirmed/refuted counts with scope) on every law.

## Session 36v (2026-10-03): THE REPLAY SHOWS THE FAILURE; THE DIRECTED RE-TEST

Tushar: "I need the replay artifact as well to visualize the game
failures. Let's have directed retest."
- Core Replays (the artifact) re-recorded for the 25 public games, two
  levels each, with the attempt ledger attached: a "Why it was lost —
  the core's own report" box; the frames where the world diverged from
  the plan's prediction marked red on the bar; on such a frame the
  dashed box where the thing was expected, the solid box where it is,
  orange what stood in the way; a lost level opens on its first
  divergence.  tools/record_replays.py attaches the ledger;
  tools/build_replays.py builds live/index.html + data from
  study/replays and files.json for the publish.
- THE DIRECTED RE-TEST (agent/core/ledger.py, play_level): a law the
  last attempt's first divergence contradicted is tested first, on
  purpose.  A key's law for a kind: that key is pressed before anything
  else on the next attempt (a thing of the kind being present).  A thing
  or ground held passable/ground/flat: it is held BLOCKING (what was
  observed) until one press through it says otherwise — the planner
  routes round it and the unblock routine's press against it is the
  re-test.  A click's relative law: dropped until a click sends again.
  The verdict is written into the next report: "re-tested 'eff
  (ACTION1, 9)' — dropped: pressed, nothing of the kind moved" (re86
  L1, attempt 2) / "restored: the law held".  Register → re-test →
  scope, as the causality note asks; the contrast (same action in
  another context) is the step still to add.

## Session 36w (2026-10-05): THE METHOD CARRIES; THE HELPER IS NOT A TARGET

Tushar: "when the system concludes out of plan and gives up, it simply
means we need to revise what we learnt, and learning may be multiple in
a game ... wa30 L2: the orange block was making our job easier by
placing crates inside the designated area and we blocked it and
confirmed there is no way out.  We did not actually understand or
evolve our understanding."  Diagnosis on wa30:1-2 (L1 won by the carry
protocol, L2 "lost in 3 actions (no_plan)"): the carry law (ACTION5
binds a crate to the body) was learned on L1 and NOT exported — the
next level began without the method, and the only readings named the
helper (the orange self-mover) as "the odd thing" to align with.  Six
general changes, each earned on that level and stated by kind:

1. THE METHOD CARRIES (fiddle.export/seed `carry_law`): the bind key,
   the bound kind's form (dims, core), the body's kind, the held colour,
   and the KIND of the thing the bound things were taken to.  The next
   level seeds the carry law when things of that form stand there; the
   protocol takes the target as the largest thing of the target kind
   (a law a win tested comes BEFORE a fresh reading's guess); a seeded
   law whose bind fails twice is dropped.
2. AN AGENT (fiddle `own_ways`/`agents`): a thing that moves press after
   press by vectors of its own — not the body's vector that press, not
   against the body (pushed), not with a neighbour (carried), and ALONE
   — travels on its own and chooses its way.  Its motion is no key's
   law; what goes with it (against it, by its vector) is in its hands,
   and those motions are not the key's law either.  Where a thing it
   carried comes to REST (it stays while the agent goes on) is recorded
   by kind (`rests`): where the helper puts them is where they go — the
   carry protocol's target when nothing else names one.
3. The protocol leaves things in the helper's hands to it (against an
   agent), counts rounds by progress (four rounds without a thing set
   down yields; a round lost to the helper taking the thing walked to
   is a round, not the level), and waits on the helper with a refused
   press when nothing of ours is left.
4. A THING IS KNOWN BY ITS FORM: a core with a one-colour border of the
   form's dims is one of the kind however the tracker cut the picture —
   three set down side by side read as one; a crate the helper set on
   the pen read as the pen recoloured.  Such things are obstacles in the
   carry plan; the cells exempt from collision are the PLACE aimed at,
   not the whole target.
5. The approach walks the whole plan (not twenty steps of it), ends on
   the body's own box beside the thing (the front row is turned by the
   facing press), and a bond left over from the test that found the law
   is let go of when the thing does not wear the held colour.
6. Two perception/planner fixes found on the way: (scene) WHAT IS WHOLE
   WHERE IT WAS, STAYS — a thing found entire at its own place is not
   dragged along by a group reading when a walker steps off it (the pen
   "moved by (0,-3)" with the body, and the body's identity was lost);
   (planner) a thing bound to a body of two members (body + front row)
   moves once per press, not once per member (the carried crate moved
   8 while the body moved 4 — every carry plan on L3 was impossible).

wa30: L1 WON 54 (was 66), L2 WON 62 (was lost in 3; the helper placed
three of six crates, we placed three — "the orange block was making
our job easier", and we let it), L3 lost in 23 (the method carries —
eight places found, a crate bound — a press the planner expected is
refused in the wall's gap; open).  The failure report now names the
agents and the method held.  Regression over the 18 plain first levels
below.

Added the same session, on L3: a press the world refused where the law
said the body moves is remembered BY PLACE in the carry protocol too
(the executor's rule) — the plan routes round it, the law stands; and
the protocol's idle allowance is six rounds (a hand-chosen constant: a
round that found a refusal is a probe along the wall).  L3 still lost:
the refusals lapse as the held crate moves along (the destination looks
different), and the kind under the crate is never seen — "step back,
uncover, look" is the next directed re-test; and the hatched column is a
kind by TEXTURE (colour 2 mixed with the floor's), which the kind-by-
colour model cannot tell from the pen's solid fill.  Bundle v37 built,
rehearsed through the Kaggle path (wa30 two levels, vc33 two levels in
600 s), delivered and pre-registered.  Snapshot /home/claude/arc3_s58.zip.

## Session 36x (2026-10-06): A KIND IS A TEXTURE; THE PATTERN GOES ON UNDER WHAT COVERS IT; AS FAR AS IT GOES

Tushar on wa30 L3: "I saw the middle line as a new type of wall, which
would not let me pass but the boxes do.  Also the understanding from the
previous game was: orange was helping us.  What would you have written
in code that, without writing the specific program, understood it?"
Four general laws, implemented and confirmed on wa30 (18/18 plain first
levels unchanged after each):

1. A KIND IS A TEXTURE, NOT A COLOUR (entities.tag_textures, fiddle.kind):
   same-colour pieces within a cell of each other are tried as one; when
   the union's cells repeat along its long side with a period of 2–8,
   three repeats or more, holes between (density 0.3–0.9), it is one
   MATERIAL — hatching, brick — a kind of its own (`body + 16`), apart
   from a solid fill of that colour.  A material carries no marks (no
   rider rule) and encloses no parts.
2. THE PATTERN GOES ON UNDER WHAT COVERS IT: two textured things of one
   colour, aligned, with a gap no longer than four periods whose cells
   all show something else, are one material whose box spans the gap.
   A KIND SEEN ON A MATERIAL PASSES IT (fiddle.read_materials): a thing
   lying within a material's box without touching its strokes stands on
   it — its kind and every colour it has worn pass that material; the
   planner lets such pairs overlap (`passable` checked at the collision,
   both orders).  So the crate passes the hatching and the body does not
   — "a new type of wall which would not let me pass but the boxes do",
   read off the first frame, before any press.
3. THE AGENT'S SIGN and WE ARE IN ITS WAY (fiddle `stalled`/`last_way`,
   carry protocol): a thing set in place that we did not set down is
   the agent's doing — its sign toward the goal (+); a thing taken out
   of place that we did not take, a foe's (−).  An agent that stands
   still while our body stands where its last own step would have taken
   it is stalled by us; the protocol steps aside, across its way.
4. AS FAR AS IT GOES (carry protocol): with no way to the place with the
   thing in hand, the search's best state is the nearest the thing can be
   brought; it is brought there and let go; a thing beyond our reach is
   left to the helper; neither is taken up again.  A carried thing
   teaches nothing about flatness (it lies over what IT passes); the
   bonds stay while it is carried and are let go of with it.

wa30 L3 now: the column is read as one material at the first look; the
crate on it says crates pass it; the body is refused into it and the
plan routes round; the first crate is handed into the column and the
helper places two.  Still lost (62–103 actions of 100): after the
hand-overs the body's laws are read through a SELECTION the fiddler
infers from the held crate's black border ("selected = black"), and
keys are then asked under that selection — the same fault that lost the
mirrored L2.  That is the next diagnosis: a thing that wears black
because it is HELD (bound to the body) is not a selection.  L4 not
reached.  Hand-chosen constants: period 2–8, three repeats, density
0.3–0.9, gap ≤ 16 cells.

## Session 36y (2026-10-06): ONE-LINE SCOPES — priors silenced where they were never meant to fire

Tushar: "so your point is that the program is not doing what it is
intended to do — that is our root cause?"  Largely yes: the organs
interfere.  Each scope below is one line, earned by a trace on wa30 L3
and stated generally; after each, 18/18 plain first levels and the five
second levels (cd82, ft09, ls20, vc33, m0r0) checked.
1. A pump at its limit refuses for ITSELF (planner): a side thing's
   retract/extend at its minimum no longer vetoes the key for the body
   — it was silently killing every key on L3 (a stale border's law).
2. A selection mark must have CHANGED HANDS — by a click or a key —
   before a mark colour becomes the selection colour (fiddle): a body's
   own dark front row moves with it every press; it is a livery.  (First
   cut counted hand-overs by key only and lost ka59, whose hand-over is a
   click; fixed the same hour.)
3. A body carrying a thing is the same body (fiddle body_moves): the
   bound thing is not a second body handed the key (no spurious
   key_selected).
4. A lone PART that moved is not the body (fiddle active): the body
   keeps its identity.
5. A body the keys move is no panel (fiddle displays; seed): a display
   is not seeded on a thing of a kind the keys moved, nor read on a
   thing that was ever active — the body was a "display" on L3 and its
   ACTION4 law was dropped as a consequence of its own turn.
6. Passable by the mover's own kind (planner): a body member of the held
   thing's colour (the front row, black like a held crate) no longer
   makes the material passable for the body.
7. THE STROKES GO ON UNDER THE COVER (entities/scene): when two pieces of
   a material are joined across a covered gap, the gap is filled by the
   period, and a material keeps its cells under what covers it — the
   planner meets the material there before the cover moves.
8. A thing that went as the key sends CONTROLLED things is no agent
   (fiddle own_ways): m0r0's second block, held back where the first
   was not, had become an "agent" and the keys' laws for its kind were
   dropped on L2 — found by the second-level check, scoped to the laws
   of things ever active.
Tried and reverted: planning the approach to the sides BEHIND the thing
(toward its place) first — right for a strait that must be pushed
through, but it cost wa30 L2 the walk round and the level; a strait
needs a plan that looks past the bind.  wa30 L3 still lost at the step
limit: the crates are handed into the column, the helper places them;
what remains is the push-vs-pull choice of side and the steps spent
probing.  wa30: L1 54, L2 64, L3 lost 103.

## Session 36z (2026-10-08): THE READER SEAM — plug-and-play for whoever makes the reading

Tushar: "we wanted the approach to be transferable — whether an LLM or
our core approach; where are we wrt that plug-and-play architecture? I
am fine with giving it a trial."  Where we are, plainly: the executor
side has been a seam all along (things -> readings in the relation
language -> the planner; laws by kind -> the model; the ledger's report
when a plan dies), but the reader was wired in by name (read_frame_goals
and the after-fiddle readings), so no second reader could be plugged.
Now it can:
- agent/reader.py: `scene_text` (the things the tracker names — colour,
  size, box, materials, marks — the laws learned so far, the frame as a
  grid, and the last failure report) is what any reader is given;
  `READING_SCHEMA` is the answer: readings {kind align|cover|equal,
  mover S<id>, target S<id>} and laws by kind (agents, helpers, foes,
  passes, refuses, hazards, carry {key, kind, target}, fiddle_first);
  `to_hypotheses` turns readings into the GoalHypothesis the planner
  takes (prio above the relations', judged by the ledger like any);
  `apply_laws` puts the laws into the model's own fields AS CLAIMS
  (model.reader_claims).  Readers: RelationReader (today's), LLMReader
  (formats the scene, calls `ask(prompt)` — the model is handed in),
  ScriptedReader (recorded answers replayed here, where there is no GPU).
- core_play2: `play_level.reader` is asked at every level opening with
  the last attempt's report; `--reader answers.json` replays recorded
  answers.  A carry reading goes to the carry protocol, not the planner;
  a claimed carry law learns its held colour at the first bind.
- Checked end to end with a scripted reading on wa30: "crates (S2) into
  the pen (S3); ACTION5 binds kind 4; target 9" -> L1 WON in 43 (54
  without a reader: the contact experiment is skipped); L2 lost by the
  step limit at 72 (64 without), the fifth crate in the helper's hands
  while we waited — the step-aside for a stalled helper is the open
  point.  Plain runs unchanged.
- THE TRIAL (study/reader_eval/): dataset.json — 44 scenes (25 first
  levels, 19 second levels), each the exact prompt an LLMReader sends;
  gold.json — ten human readings by colour (claude/human-readings.md);
  reader_trial.ipynb — a Kaggle notebook that loads a quantized Qwen3.8
  from Kaggle Models, answers every scene, records seconds per call,
  scores the gold matches, and saves reader_answers.json — which
  `--reader` replays into the agent here.  What it answers: (a) can
  the model make the reading a person makes, in our language; (b) the
  cost per call against ~245 s of thinking per game in the rerun; (c)
  its readings on the six games the core has never won.  Tushar runs
  it (GPU); I read the answers and replay them.

## Session 37 (2026-10-13): THE NORMALISATION PROGRAM — solve everything, then take the organs away

**The LLM trial is stopped** (Tushar: "let us stop this").  Record of where it
got: the leaders' model (Intel W4A16 AutoRound of Qwen3.8 Flash Next, 125B with a
51B-entry n-gram table) loads on Kaggle's RTX Pro 6000 only with three repairs —
auto-round from a wheels dataset (internet is refused with that GPU), an explicit
device map (transformers 5.16.1's `auto` planner exiles every layer after the
n-gram layer to the CPU on a single GPU), and the cache warm-up disabled (222k
tensors, most tiny; the warm-up's one block starves them) — and had not yet
produced a single answer after four sessions.  The seam (agent/reader.py, the
dataset, the scorer, the scripted reader that wins wa30 L1 in 43) stays; the
trial is parked, not refuted.

**The program.**  Three tables, run every day, each row a general statement:

1. *The ladder* — every public game, as far as the core climbs at 300 s a level
   (tools: /tmp/ladder/lane.sh; two lanes; started this session).  The first
   measurement of the whole suite since the gauge.  Each lost level gets a
   diagnosis in the failure atlas before any fix; fixes land only as general
   statements and only with the plain suite held (regC/regD).
2. *The gauge* (tools/invariance.py, extended) — the same game under a different
   skin: colour derangement (done), mirror / quarter turn (done), **background
   swapped** (new), **a novel thing added** (new: a static block of an unused
   colour in an empty region — a perception test: does the core keep its reading
   when the scene has one more thing in it).  Mechanic twists (ally becomes
   enemy, support becomes damage) cannot be made by a frame transform; the
   game's own later levels ARE those twists (wa30 L2 ally, L3 wall-the-boxes-
   pass), so the ladder is the mechanic gauge until a concept simulator exists.
3. *The ablation* — every organ behind a switch (NO_TEXTURE exists; the rest
   to add), the table organ × game: a level that only one organ wins is a
   curve-fit suspect; an organ that wins nothing is cut.  Then the reduction:
   organs removed one by one while the ladder and gauge hold — what remains is
   the claim of the paper: perception + a few skills + the ledger of what
   worked, not forty priors.

The order of the day: ladder first (measure), then diagnoses from the bottom
of the ladder up, the gauge and ablation re-run after each change.  Deadline
Nov 2 (code) / Nov 8 (paper): the tables are the paper's results section.

### 37b (2026-10-13): THE TWIST SUITE — the same game, a different law

Tushar: "you have to create games with different mechanics ... ally becomes
enemy, supporting becomes damaging ... there has to be an emergent phenomenon:
you get feedback, self-correct and predict correctly."  tools/twistgen.py
writes four arcengine games (environment_files/tw01–tw04), three levels each:
L1–2 teach a law, L3 wears the same skin and breaks it — tw01 the helper
pushes blocks toward the pen, then away; tw02 a floor colour refills the
budget bar, then drains it (death); tw03 a line the player cannot cross while
blocks slide through, then the player crosses it; tw04 block-onto-pen ends
the level, then the block starts on the pen and the mark is the goal.  Every
level is proved solvable by an A* over the engine itself (the solution
length is the baseline).  The governing law is in metadata.json for the audit
only.  Measured: not the win but the UNLEARNING — the ledger must show the
carried law predicting wrong, the contradiction re-tested, the law dropped or
re-scoped, the level won under the new one, with no line of the core naming
the twist.

Baseline (core alone, 180 s a level): L1 4/4 (tw01 58, tw02 34, tw03 7,
tw04 19 actions); L2 1/4 (tw02 11; tw01/tw03/tw04 lost "no_plan" — a new
layout of the SAME law defeats the carried reading before any twist);
L3 0/4 — only tw02 reached its twist and died twice (12, 15 actions).
First diagnosis, from the failure report: on tw02 the core takes the HUD bar
(kind 10, shrinking by one cell per action) for the mover — "S3 (kind 10)
expected at (28,28), elsewhere; in the way: S4 (kind 12)" — and after the
death doubts the HUD's law, not the pad.  The self-correction machinery runs,
pointed at a misread body: perception first, as always.  Next: that
diagnosis (a thing that shrinks in place under every key is a meter, not a
mover — stated generally, confirmed on the plain suite), then the three L2
"no_plan" losses, then the twists themselves.

### 37c (2026-10-13/14): THE FIRST TWIST UNLEARNED — four general statements from one synthetic game

tw02 (the pads) traced press by press.  What the first failure report called
"the HUD taken for the mover" was three faults in the tracker and one in the
evidence, each fixed as a general statement and none naming pads or bars:

1. **A thing in its own colours before a thing recoloured** (scene.py): the
   pad under the arriving walker was read as "the pad recoloured to the
   walker's colour, staying put" (a staying reading outranked a moving one
   at equal cells), and the walker, its cells taken, slid onto a far
   look-alike — the last cell of the budget bar.  Now a reading that keeps
   every colour outranks one that must suppose a recolouring (a 0.02
   tie-break, below any real cell of difference), and "whole where it was"
   requires its own colours.
2. **Cut cleanly from one side, it is still the thing** (scene.py): the
   budget bar drained from 40 cells to 4 was reborn under a new number each
   time, so it never accumulated the presses that make a counter.  What
   remains of a thing at its place, in its colour, as the thing with one
   side cut away, is that thing retracted — however much was cut.  With
   this the bar is read as a COUNTER and then as THE CLOCK (16 cells a
   press) on the first level, and carried.
3. **Pushed, not hidden** (scene.py, found on tw04): a block the walker
   pushed was read as "hidden under the walker" (the tile-under-walker
   rule) and a new block born ahead at every press — identity churn, laws
   of (0, 8).  A thing whose moved reading lands on cells nothing else
   explains did move; hidden-under is a reading only when where it would
   have gone is someone else's.  tw04 L2 went from lost to WON 119, L3 (the
   goal twist: the block starts on the pen, the mark is the goal) WON in 14
   — the search's own optimum.
4. **The drain is the refill's twin; this level's evidence outranks the
   carried** (fiddle.py): an arrival on which the clock lost more than two
   presses' worth is a drain of the clock, as an arrival on which it rose
   is a refill — and a floor colour stepped onto alone is credited with the
   jump (the "floor is not the refill" exclusion applies only when
   something else was arrived on too).  A drain is a hazard to stand on
   unless this level has shown the kind refilling.  Same principle for
   deaths: carried death/alive counts are priors, consulted only for kinds
   this level has not judged.
   Result: tw02 L3 WON in 18 — the carried belief "pad refills" walked the
   body onto the pad (plan of 10 through it), the clock's fall on arrival was
   read as a drain, the pad became a place not to stand, the level restarted
   with the clock full, and a 12-step detour (the optimum) won.  Predict,
   contradict, re-scope, win: observable in the trace, no line naming the twist.

Twist suite now: tw02 3/3, tw04 3/3, tw03 L1 only (L2: a slide puzzle in a
new layout, no plan — diagnosis next), tw01 L1 LOST (regressed: the helper
is now read as an agent and its blocks left to it — correct — but the core
then sits IN the pen pursuing "reach the pen" and blocks the helper's
deliveries; the stalled-agent step-aside of wa30 L2 did not fire: diagnosis
next).  regC (18 plain first levels) restarted on this code; the ladder's
two lanes still running.  Snapshot /home/claude/arc3_s62.zip.

### 37d (2026-10-14): THE LADDER, WHOLE — 28 of 183 levels; the walls by kind

Full table and reading in claude/ladder-2026-10-14.md.  18 games yield L1, 6
a second level, ft09 complete (6/6); seven yield nothing (sp80 sc25 sk48
dc22 g50t re86 sb26).  Walls: W1 no reading (9 levels — the largest, several
with `keys known []` on a carried level); W2 a reading but no plan (6); W3
the law lied — "the mover stayed where the plan moved it" (7, five of them
the never-won games); W4 died (3).  Agents organ ablated (NO_AGENTS=1): earns
wa30 L2 only, blocks nothing, over-reports on five levels.  tu93 regression
from the push rule fixed the same hour: "what lands where the walker just
stood is uncovered, not moved" (tiles stay; pushed blocks move) — 18/18
plain held.  Order: W1 on carried levels first (one fault, up to four
levels), then tw01/tw03 L2, then cn04 L2, then sk48.

### 37e (2026-10-14): "WE SIMPLY DISCARDED EVERYTHING" — where the carry forgets, and three repairs

Tushar: "the code itself is not sufficiently well written and we simply
discarded everything on any new observation."  Checked against the traces of
the W1 levels (no reading on a carried level):
- tn36 L2, lp85 L2, s5i5 L2: `carrying to L2: {'eff': {}, ...}` — the carry
  is EMPTY after wins by clicks.  export() says it in a comment: "a click is a
  place, not a law".  Every button game forgets its buttons between levels.
- tn36's win by the program organ (launch, journey, strip of toggles) leaves
  nothing in the carry at all: organ state is not exported; L2 re-fiddles 40
  clicks and never finds the strip.  Same for leap (lf52), legend, table.
- bp35 L2 "lost in 1 action": not forgetting — a walk search took the
  level's whole clock twice (plan_general's 90 s, 16k states) and the
  readings never got their turn.
Built (each general; 18/18 plain held; wa30/cd82/m0r0/vc33/ka59 climbs and
ft09 6/6 unchanged):
1. **A button carries by its kind** (fiddle.click_under, export
   "click_laws", seed): a click on a thing of kind K that moved things of
   kind M by v is the law "K sends M by v"; on the next level every thing of
   kind K is presumed such a button, provisional until the first click.
   (lp85 L2 is NOT this case: its second level is the track permuter with
   different kinds — a new mechanic, open since session 24.)
2. **This level's evidence outranks the carried** (37c): deaths and the
   clock's refills/drains — the carried counts are priors, not votes.
3. **No search takes the level's whole clock** (planner.time_left): a
   search gets at most a quarter of what is left; bp35 L2 now tries its
   readings (and fails on them: a W2, "collect 3 like things" with laws that
   move the body along one row only).
Next in the same vein: the ORGAN CAST — a level won by program / leap /
legend / table / display exports which organ and its cast by kind (the
launch's kind, the traveller's, the toggle colours, the strip's form), and
the next level opens by mapping the cast onto its things and running the
organ's protocol first, re-fiddling only what the cast lacks.

### 37f (2026-10-14): THE JOURNAL — the whole play remembered, the highlight carried (Tushar's micromouse)

Tushar: "remember the entire play of the previous level compressed and
extract the highlight on what we actually did; the micromouse analogy: if we
ever play twice we simply figure out the shortest route; nothing is lost —
and reinterpreting the learning is the intelligent part."
agent/core/journal.py: every press journaled (key, the core's phase, what
changed, the frame it led to); at the attempt's end the HIGHLIGHT: the route
after the last reset, the safe prefix (before the first divergence of the
plan from the world; on a death with no divergence, the whole route to the
brink), the fatal pair (the frame and the press that killed), the story (the
reading pursued / the organ), and where the changes came from by phase.
Printed at every level's end ("what we actually did") and carried.
- THE SECOND RUN: a retry of the same level after a death (its first frame
  is the one the last attempt began on) walks the safe prefix again, each
  press checked against the frame it led to last time — the world keeps its
  own clock (tu93's oscillating tiles), so the replay ends at the first
  press that leads elsewhere and the core looks anew from there; the fatal
  press is refused at the frame it killed on.  A death now always earns a
  second attempt (before: only when a hazard KIND was named).
- tu93 L2: lost (died, 8) → replay to press 4, "ACTION4 from this very frame
  was the fatal press" refused eight times in planning, WON in 36 (plan
  3/3).  29 of 183 levels.  sp80 L1 and ka59 L3 die the same way twice: the
  replay ends at once (a world on its own clock) — their deaths need a kind
  or a timing, not a route.  18/18 plain held; the climbs unchanged.
Next: the story REINTERPRETED on the next level — the organ cast (program /
leap / legend / table / display by kind), and the journal's "taught" lines
as the abstraction's input (which phase produced the knowledge is itself a
lesson: a level won by fiddling 8 presses then one plan is a level whose
keys matter; a level won by clicks on one kind is a button level).

### 37g (2026-10-15): THE CAST — the organ's parts found again by kind on the next level

fiddle.cast(): what the level's mechanisms were made of, by kind and form,
no place: the launch button's kind and box/size and the kinds it sent
travelling, the colours the toggles flip between, the leap's pitch.  Carried
with the lesson.  core_play2: the protocol that won the last level now opens
the next with its cast reinterpreted — a launch candidate of the launch's
kind (most like it in box, then size; up to three, one press each), a toggle
candidate of the flip colour (one click), then the organ.  And the carried
opening itself was unlocked for levels won by clicks: an empty `eff` used to
skip the whole opening ("keys known []" on tn36/lp85/s5i5 L2); buttons, cast
and template now open it too.
tn36 L2 traced: the cast finds the launch's kind (9) but on L2 the launch is
a DIFFERENT button of that kind (38 cells, same 9x9 box; the 69-cell one is
there too and is no longer a launch), and the traveller (kind 15) is not on
the board at the start — something must put it there first; after the
fiddle's launch the cast's toggle click lands (colour 1 marks, strewn 44
times, which the fiddle never probes) and the program organ runs — and
fails: "the launches do not agree on a legend", because L2's panel is a
GRID of slots (4 columns x several rows; L1 had one row of 5) and
read_panel reads a strip along one axis.  The reinterpretation needed is
in the organ: a panel is a lattice of slots read row-major.  An organ
build, specific in appearance, general in statement; deferred behind the
transfer work.  18/18 plain held; no new second level from the unlock
alone (lf52 r11l s5i5 su15 bp35 lp85 cn04 ls20 tr87 ar25 L2 as before).

### 37h (2026-10-15): THE CLAIM LEDGER — a prior is a claim with a track record

Tushar: "the claim ledger first ... we just need to win every level and then
win also on new mechanics."  agent/core/claims.py: a standing presumption at
a decision point is a CLAIM, consulted through `claims.on(name)`; a plan is
built consulting claims; a step that holds credits every claim the plan
consulted, a step the world leaves strikes them; a claim struck at least
twice and more than it held on THIS level is SILENCED for the rest of the
level (the decision point takes the other branch — the next plan is the
directed re-test); the record is carried by game as a prior, consulted
only for claims this level has not judged.  CLAIMS_OFF=a,b (the ablation
table by hand) and CLAIMS_FROZEN=1 (the pre-ledger core) for measurement.
The eight one-line scopes of session 36y are the first claims:
pump_limit_refuses_only_itself, passable_by_kind, texture_carries_no_rider,
key_sent_is_no_agent, mark_must_change_hands, moved_kind_is_no_panel,
lone_part_is_no_body, carried_is_no_second_body.  Each level's end prints
"claims this level: name: held h, struck s" — the self-programming is
visible.  18/18 plain held, the climbs and ft09 6/6 unchanged; tw01 (the
ally) now L1 WON 100 and L2 WON 159 (was L1 lost); the ledger's first act:
m0r0 L3 silenced carried_is_no_second_body (held 0, struck 2) — the right
direction for a two-body game (the level is still a W2).
Known weakness, by design for now: attribution is by the plan as a whole
(every claim it consulted is struck together); the directed form — silence
ONE suspect, replan one step, contrast — is the next refinement, and the
ablation table reads off the same records.

### 37i (2026-10-15): THE DIRECTED RE-TEST; THE PANEL AS A LATTICE; THE LEGEND BY EXPERIMENT

1. **The directed re-test** (claims + core_play2._record_div): on a
   divergence, the claims consulted for THAT step are found by re-running the
   step's prediction from the state before it with a fresh consultation
   record; each is silenced in turn and the step re-predicted; the claim
   whose silence makes the prediction match the world is struck twice
   (silenced) and the rest credited.  When nothing explains it and something
   stood where the mover was sent, no claim is struck — the lesson is that
   thing's (a wall, a kind not known to block); only a divergence with
   nothing in the way and no claim to explain it strikes them all.
   tests/test_claims.py (4), all 18 unit tests green.
2. **The panel as a lattice** (program.read_lattice): a panel is a region of
   one colour holding like units at one pitch along each axis; the columns'
   pitch from the largest group of like units, the rows' from the marks'
   own groups, tiles centred on their units, read row-major; a row of slots
   is the lattice with one row.  tn36 L1 reads as (1, 5), L2 as (3, 4).
   The strip reader stays as the fallback.
3. **A journey may be a prefix** (program.correspond prefix=True): twelve
   like slots and a journey of three — the journey ended early; the slots
   beyond it are left unread.
4. **The legend learned by experiment** (program_protocol): with one form
   known, the organ changes slot 1 (one mark flipped) and launches, and the
   new journey says what the new form means — a few tries.
tn36 L2, where it stands after all four: the lattice is read, the journey
is a prefix, the experiment runs — and the journey does not change when
slot 1 changes.  The human reading (claude/human-readings.md) says why:
"L2+: a DEMO panel teaching opcodes, six-toggle slots" — the legend on L2 is
DRAWN (the left half of the board), as ft09's is; the program organ should
read a drawn legend before experimenting.  That is the composition of two
organs (legend reader + program writer) and is the precise next statement
for tn36; it is not written in.  18/18 plain held; climbs and twists as
before (tu93 L2, tw01 L1–2 kept).

### 37j (2026-10-15): THE WIN READ HONESTLY; THE REFEREE ON THE WHOLE CODE

1. **The win read honestly** (play_game): a level won is a lesson only if
   the core knows WHAT it won by.  Before, the story carried was the reading
   PURSUED; on tw03 L1 the core pursued "the odd thing C14" and won because
   the state came to satisfy "the odd thing C8" — the win frame is the first
   frame of the winning press (obs.frame[0]), read with read_components, and
   every first-frame reading is checked against it (colours moved since the
   first frame are allowed; the target itself pushed off its own cells is
   excluded).  The reading the win state satisfies is the story carried —
   "the win read honestly: pursued X, the state satisfies Y" in the trace.
   No constant chosen.
2. **The referee on the whole code** (the sealed set and the sim holdout,
   counts only): sealed 1/6 (sx11 L1–2, as before); holdout 10 of 23 games
   with L1.  The plain 18/18 held; climbs tu93 1-2, wa30 1, cd82 1-2, m0r0
   1-2, vc33 1-2, ka59 1-2, ft09 6/6; twists tw02 3/3, tw04 3/3, tw01 L1–2,
   tw03 L1 — all as before.  Nothing moved on the referee: the sessions
   37e–37j repaired the CARRY (what is kept between levels) and the carry
   shows on climbs (tu93 L2); the referee's games stop at L1, before any
   carry.  The next work is the first look (W1) and the law that lied (W3).
Snapshot s68.

### 37k (2026-10-15/16): tw03 3/3 — THE SLIDE PUZZLE AND ITS TWIST, seven general statements; two regressions found by bisection

Order of work set by Tushar ("win every level, then win on new mechanics"):
tw03 L2 (the slide puzzle, a W2 on the twist suite) diagnosed on its trace,
one statement at a time, each general, none naming tw03:
1. **A divider runs edge to edge** (perception.detect_playfield): a uniform
   line that splits off a HUD runs the whole frame; a line that ends at the
   room's walls (the strips stripped first, of another colour) is a thing
   inside the room.  The pen beyond the line had been "HUD" on L1 and L2 and
   never read (L1 won by luck).  Surveyed on all 25 public first frames:
   only dc22 (never won) changes besides tw03.
2. **The winning press is a press like any other** (step): the engine
   returns the level's last frame before the next level's first; the press
   is observed on it before the next level is looked at.  L1's only push was
   the winning one — L2 opened not knowing a push exists.
3. **A shove that came to rest against something slid until stopped**
   (fiddle.push_stopped / push_run): the run seen is a law of its own only
   when it stopped with open ground beyond; concluded only when EVERY run of
   the kind was stopped and one was longer than the pusher's own step (the
   first cut, "any run stopped", made ka59's frames slide to the walls and
   lost ka59 L2 — caught by the climb, narrowed, ka59 L2 back at 56).
4. **A mover that stayed with something in its way: the lesson is that
   thing's** (ledger.doubts_from): the key's law is not doubted — it was
   never tried against open ground there.
5. **The template judges "the same target colour" on the note, never
   waived** (_template_reading), and among this level's readings of the
   winning relation those with the same MOVER kind come first — it had
   pursued "the odd thing C9", the stopper, as the template for the pen.
6. **A pushable thing is not a place to arrive at**: it stays solid even as
   the target (two "lifted" rules and the planner's "target cells are no
   obstacle" rule now except pushable kinds), and a reading aimed at a THING
   is measured to where the thing stands in each state (t_sid_ in dist and
   the "reach" goal).  "Reach the block" had been satisfied by shoving the
   block into the far corner, the player taking its place — the level lost
   right there, before the template got its turn.
7. **A wall carried from the last level is a theory of that level** (the
   carried-wall re-test in pursue): when the search ran out of ROOM, not
   time, each carried ground refusal is lifted in turn and the way through
   it planned; a plan found is followed, its first step into that colour the
   re-test, read by the observer as the ground allowing or refusing.  Each
   colour once a level; a colour refused on THIS level too is no theory of
   another.  tw03 L3 (the player may now cross the line) WON in 8.
Two bugs on the way: the MIRROR thing-law (thinglaws) never counted a frame
against it unless a wrong thing stood AT the image's place, so "the pen
mirrors the player" held on three coincidences and a phantom image rode every
plan (fixed to the docstring's rule); the planner spent a fifth of a search
on state signatures (sorted, stringified) — now one hash of the state's
items, and boxes of frozensets remembered.  tw03: 7 / 52 / 8.

MEASURED, honestly: the second-level lane (tn36 lp85 s5i5 bp35 lf52 su15
r11l ar25 ls20 cn04 tr87) is 0/11 — the carry repairs of 37e–37j moved tu93
L2 only.  bp35 L2 still spends its clock thinking (profiled: succ ~3 ms a
state, push_move and _sweep the cost).  cn04 L2 and ls20 L2 were WON in
September (sessions 33, 36n) and are lost now — found only because the
failure reports were re-read; the climb list did not include them.
BISECTED over the snapshots: cn04 L2 flipped between s51 and s52 (Sep 26),
where the only core change was the wall clock (s51 ignored --seconds and
thought 600 s; the current code at 600 s still loses it, so more changed
since); ls20 L2 still won at s62 (Oct 13) — lost by this week's sessions,
bisection running over s63–s67.  Rule from this: EVERY level ever won is in
the climb list from now on (cn04:1-2, ls20:1-2 added to chain7).
Later the same night (37k continued):
8. **A pushed thing's moves are not the key's law** (fiddle.export): a
   thing that moved under a key only on presses where the mover came
   against it (pushed_count ≥ its presses moved) carries no law of that
   key — tw04 L3 had planned the block walking to the mark by itself
   (ACTION2 "moves kind 14", from L2's pushes).  tw04 L3 WON 26 (plan 13).
9. **A shoved thing stops at ground never seen crossed** (make_successors
   shove_walls): the body learns a wall by bumping (one press, nothing
   lost); a thing pushed into unknown terrain may be lost for the level
   (tw04 L3: the block pushed to the border, the mark out of reach).  Terrain
   no shove has slid over and the body has not walked on refuses a pushed
   thing in the plan; when that leaves no plan, the way across it is tried
   as the re-test (one push the experiment), as with carried refusals.
10. **The thing aimed at, by its colour, never the body standing on it**
    (t_sid_): the player had walked onto the mark, and "align(block → mark)"
    then tracked the player.
11. **A pushed thing is seen whole where it went** (scene.py, the
    pushed-not-hidden statement of 37c): a reading that matches PART of a
    thing elsewhere is no push.  THIS was the ls20 L2 regression — found by
    bisection (s61 won, s62 lost; A/B by module: scene.py; A/B by hunk: the
    third statement of 37c): a glyph "moved" nine rows and five columns on
    five of its eight cells.  ls20 L2 WON 47 again.
12. The "odd thing" size cap (relations.py, 60 cells) is now relative: no
    larger than four of the mover (hand-chosen; the pen of 64 cells).
13. push_run's "slides until stopped" narrowed (see 3): the first cut lost
    ka59 L2; caught by the climb.
THE CLOCK, FOUND: since 36o a level turn is TURN_FRACTION (0.25) of the
game's remaining clock — with `--seconds 300` every ladder and regression
level has had ~75 s of thinking, not 300.  cn04 L2 was won at s51 (600 s
turns) and lost from s52 on; the s52 code with full turns wins it (65), the
current code with full turns does NOT (15/33, no_plan) — a second, code
regression between s52 and now, bisection queued (TURN_FRACTION now an env
override for such runs).  wa30 L2 (won 64 on 10-14 with the agents organ)
is lost since this morning's chain — open.  tw01 L2 open: the two blocks
read as one 32-cell thing, the story carried from L1 is "picture equal" (the
player as mover), so no block→pen reading is pursued.
Chain on the settled code: 18/18 plain; tu93 L2, cd82 L2, m0r0 L2, vc33 L2,
ka59 L2, ls20 L2, ft09 6/6; tw02 3/3, tw03 3/3, tw04 3/3, tw01 L1.
Snapshot s69.

### 37l (2026-10-16): THE LADDER WITH FULL TURNS; three regressions found and repaired by bisection

The sandbox freezes between turns: a run only advances while the session is
in a turn (the overnight ladder ran seven minutes).  Re-run attended.
THE LADDER, 300 s A LEVEL (claude/ladder-2026-10-16.md): 31/183 from 28.
Nothing is a time wall — every level lost at 75 s is lost at 300 s, nearly
all in seconds (no plan); s5i5 L2 and re86 L1 (a never-won game) are won by
the 37k statements and win at 75 s too.  The quarter-clock rule stays for
the harness.
REGRESSIONS, bisected over the snapshots and repaired generally:
1. wa30 L2 (lost since 10-15 morning): "the win read honestly" (37j) had
   invented a story for a level won by an ORGAN with no reading pursued
   (carry) — "the odd thing C2" read off L1's end state — and L2 died
   pursuing it.  A level won by an organ carries the organ as its story; a
   reading the end state happens to satisfy is not what won it.  Back at 64.
2. cn04 L2 (won in September, lost at s64): the first tracker statement of
   37c — "a thing in its own colours before a thing recoloured" as a 0.02
   penalty on the placement score — sent the selected piece (drawn in the
   selection colour) to a look-alike across the board, fifteen times a
   level.  Replaced by an explicit placement ORDER among readings of one
   score: whole where it was in its own colours (2); arrived by a step in
   its own colours (1); stayed and shows another colour (0.5); the rest.
   The pad under the arriving walker still yields to the walker (tw02 3/3,
   22 actions) and the selected piece stays the piece (cn04 L2 WON 65).
3. Two crashes: a verbose print sorting keys of mixed types; the slide
   check `_same_way` on a non-translation law (cn04's retract clicks).
   nudge_toward now uses only translation laws of the selected keys.
4. cn04 L2, honestly: every tracker variant that wins it loses something
   older.  Withdrawing the 37c own-colour tie-break and offering "recoloured
   where it stands" beside a whole shift elsewhere wins cn04 L2 (65) and
   keeps bp35/tu93/ka59 L1, tw02, s5i5 L2, wa30 L2 — and loses tu93 L2 (the
   walker that left a tile); the tie-break alone with the option loses s5i5
   L2; every placement ORDER tried lost tu93 L1 or bp35 L1.  The tracker is
   left as s69 had it (tu93 L2, s5i5 L2 kept; cn04 L2 lost as it has been
   since September) and the finding is written at the tie-break: the
   statement that tells the piece drawn in the selection colour from the
   walker that left a tile is still to find.  The referee that found each
   level: tu93 L1, bp35 L1, ka59 L1, tw02 1-3, plus cn04 L2, tu93 L2, s5i5
   L2, wa30 L2 — eight levels, ten minutes.
5. EVERYTHING A KEY MOVES MOVES ON ONE LATTICE (planner.Fixed.observe): a
   displacement that is no multiple of the step the key's held laws take is
   a misreading of the tracker, not a law (wa30 L2 under the variant above
   carried "ACTION2 moves kind 9 by 7").  Kept: it is true on its own.
Chain on this code: 18/18 plain; tu93 L2, wa30 L2, cd82 L2, m0r0 L2, vc33
L2, ka59 L2, ls20 L2, ft09 6/6; s5i5 L2 72, re86 L1 28; tw02 3/3, tw03 3/3,
tw04 3/3, tw01 L1.  Snapshot s70.

### 37m (2026-10-17): tw01 L1–L2 — THE PUSHED BLOCK, nine general statements; one regression (ls20 L2) and its repair

tw01 (twist of the block-pushing room): two like blocks, a helper that walks
on its own, a player who must push the blocks into the pen.  Won L1 (65) and
L2 (49) on these statements, none naming the game:
1. FAILED READINGS ARE CARRIED (core_play2 failed_readings, by rule / mover
   kind / target kind): a reading that "arrived, nothing happened" goes
   LAST among the pursuits of the next attempt and the next level.  It
   orders, it does not forbid (the tw01 reading "the player fits the
   helper's shape" had been walked to its end seventeen presses an attempt).
2. A PUSHABLE KIND IS A MOVER TOO (actuated): for align/inside readings and
   at most four instances (the pushed block goes to the pen; tu93's forty
   tiles do not) — and the AGENTS (things that travel on their own) are
   demoted as movers: nothing of them is ours to send.
3. SET ASIDE: an align/inside reading whose TARGET is pushable is not a
   goal (one block is not sent onto the other).
4. INTO THE ODD THING (relations): a filled larger unique thing is read as
   a container (inside); the goal is all mover cells within its box.
5. THE PUSHED THING TEACHES NO TERRAIN (fiddle): a kind that was pushed is
   not blocking, not passable, not flat; a part pushed off is not flat
   ground that reappeared.  (Before, "block passable" was learned from the
   block's own displacement.)
6. "SLIDES UNTIL STOPPED" only when every push-run stopped and one exceeds
   the pusher's step (push_run); ka59 L2 kept.
7. ONE OF SEVERAL LIKE THINGS IS IN; THE NEXT IS PURSUED (h._members_done,
   redo_pursuit): the reading continues with the next member instead of
   failing at "arrived, nothing happened".
8. SPLIT BY MOVE (scene._four_pieces, strict): two touching like blocks
   read as one thing are two when a press moves one piece — equal size,
   one piece on the old cells, the remainder translated by one vector.
   (The loose split lost cd82 L1; the strict one keeps it, 65.)
9. "pushed, not hidden" requires the thing seen WHOLE where it went
   (ls20 L2 had regressed on the 37c third statement).
REGRESSION: ls20 L2 (47 → lost, no plan).  Bisected by env switch to (1):
the failed-readings sort ran BEFORE the cut to max_single, so two "reach
the odd thing" readings that had once failed on L1 fell off the end behind
readings the reader ranked lower — and the walk that shows the display's
mechanic was never taken.  Repair: the cut is by the reader's rank, the
failed go last WITHIN it (the record orders, it does not choose).  Also:
a reading the win vindicated (won_hyp) is struck from the record.  ls20 L2
back at 47; tw01 L1–L2 hold.
Noted: tw01 L3 won once (9 actions, plan 3/3) and lost in later runs — the
enemy helper's timing; to diagnose.  s5i5 L2 is timing-sensitive under
machine load (three runs at once lose it on any snapshot), not a code
regression.  Debug switches left in, all off by default: NO_SPLIT,
NO_FAILED_ORDER, NO_SET_ASIDE, NO_INSIDE, PASS_PUSHABLE, FLAT_PUSHABLE,
BLOCK_PUSHABLE, OLD_SIG, PLAN_TRACE, SPLIT_DEBUG, CONTACT_DEBUG, FLAT_DEBUG.

### 37n (2026-10-17): bp35 L2 — WHERE THE CLOCK WENT; the split press (tw01 L3) left open

bp35 L2 at 300 s: "lost in 1 action, no plan, readings tried 0".  Timed
(walk/leg prints) and profiled (cProfile, 434 s):
1. A COLLECT READING WITH NO WAY TO ANY MEMBER IS NOT WALKED: the legs
   (one search per member, 26 s each, all None) were followed by the walk
   to the nearest — three clock modes, three more exhaustive searches of
   the same space, 105 s.  Two such readings took 250 s of the 300.  Now:
   one feasibility search with every member out of the solids (_leg_any,
   a 'no' there is a 'no' for each member); none → the reading is left
   unplanned (not refuted: a way may open), nothing walked.  Legs per
   member are computed only when a way to some member exists.
2. THE SWITCHES ARE READ ONCE: `os.environ.get("X_DEBUG")` is a Mapping
   call (encode + lookup); one stood inside the planner's successor loop
   over every state entry — 20 million calls, 60 s of the level.  Every
   module now reads `_ENV = dict(os.environ)` at import.  (The profile's
   next items — succ 0.6 ms own time per call, push_move 1.2 M calls,
   isinstance 221 M — are the planner's own cost; left.)
   After: bp35 L2 reaches its plans and dies at ACTION4 after moving
   blocks (12 actions) — the level's mechanic (a vertical corridor, blocks
   removed or dropped by clicks, the screen scrolling) is still unread;
   a frame-by-frame study is the next step there.
3. tw01 L3, diagnosed: the two blocks start touching and are read as one
   thing (S3, 32 cells); the first push splits it (37m's split-by-move,
   fires) — but the OBSERVER reads the parent's loss as the law 'ACTION4
   retracts the block's bottom by 4' and the plan moves the pair.  Two
   repairs tried, both lose tw01 L2 (49 → lost) and must be wrong as
   stated: (a) re-reading the frame before with the piece as its own thing
   where it stood (the piece then moves with the player on its first sight
   and is taken into the player's body); (b) no law for parent or piece on
   the split press (L2's plan then lacks what the retract law gave it —
   the push's refusal, read through it).  Reverted; the statement for the
   split press is still to find: what the baseline learns from the retract
   law that the push laws should carry instead.
Referee after: quick3 6/6, tw01 L1–L2, ls20 L2, cd82 1-2.  Snapshot s72.

### 37o (2026-10-17): WHAT THE WRONG LAW STOOD IN FOR — tw01's 'retract', by contrast

Tushar: a wrong law that works is fine; the job is to re-evaluate what
gave the same success and find the explanation with its own reasons.
Applied to tw01 (the baseline carries 'ACTION3/4 retracts the block by 4'
— the observer's reading of the pair's split — and wins L1 65, L2 49;
without that law (SPLIT_NOLAW=1, the switch is in) L1 is won by explore
at 71 and L2 is lost).  The contrast, L1 second attempt, same frame:
  BASELINE: after the split, the plan for "the pair into the pen" finds
  NO PLAN (the retract law shrinks the mover; 2662 states, 4.2 s) → the
  core fiddles again (the fourth-direction check), learns kind 6 (the
  helper) as BLOCKING, re-reads the frame, and the same reading — its
  mover re-resolved to one half (S6) — plans and wins.
  NO LAW: the plan for the same reading finds 15 steps at once, executes
  with kind 6 still unknown as a wall, diverges at step 2 (the mover went
  further than predicted) and step 8, finds no plan, and the attempt is
  spent on replans; the win comes by explore.
So the law bought two things: (a) the pair's plan was refused right after
the split, and (b) the time that refusal gave went into a fiddle that
learned the helper blocks.  The reading itself was never wrong — it is
about the KIND, and its mover re-resolves to a half.
Tried as the explanation: (a) alone — "a press that split the mover
withdraws the reading; the frame is read again" (SPLIT_WITHDRAW=1, kept
off): L1 won by explore 71, L2 lost — the withdrawal removes the reading
that later wins.  Not it.  Still to test: (b) — a plan is not executed
through a kind whose blocking is unasked when the plan passes beside it
(the planner's "kinds never asked" list exists for the failure report;
the fiddle could be asked first), and (c) the planner's push prediction
on the step-2 divergence (the mover went FURTHER than predicted: a push
read as stopped that was not).
The environment was reclaimed between turns (the sandbox is rebuilt from
arc3_s72.zip + the 25 public games from Tushar's machine + twistgen;
python 3.13 venv; plays identically: tw01 65/49, ls20 28/47).

### 37p (2026-10-17): THE SPLIT PRESS, READ OFF THE FRAMES; the ledger carries what the plan knew

Tushar: keep the gameplay so the insight comes off the record, not a
re-run; and the loss was "no chances left after the fiddle" — the plan
spent the attempt before the knowledge was there.
1. THE LEDGER CARRIES WHAT THE PLAN KNEW: every divergence now records
   the beliefs the step was predicted from — blocking, pushable, push
   vectors and stops, the agents, the kinds on the frame never asked, the
   plan's length and search — printed in the failure report ("the plan
   knew: …") and written into the replay's ledger.  The contrast of 37o
   is now readable from the journal.
2. THE SPLIT PRESS, FROM THE DUMPED FRAMES (tw01 L1 second attempt, frames
   151→156): the pair (24,28)-(31,31) was split by the HELPER (kind 6, an
   agent) walking right into its lower half — the player was elsewhere,
   moving up.  Three presses later the player pushed the upper half right
   while the helper pushed the lower half right, both by (0,4); the
   tracker then gave the upper half's name (S3) to the LOWER half's new
   place, 8 cells away instead of 4, and every plan after that reasoned
   about swapped things.  The 'retract' law never entered this; it only
   kept the first plan from being made.
   So the retract law stood in for: the pair's reading planned through a
   thing the helper had just split, with the halves' names unstable.
3. Tried: reading the frame before with the piece as its own thing where
   it stood (fiddle, SPLIT_READ=1, kept off): the push is the piece's,
   the parent gets no law — correct in itself, but L2 is lost the same
   way, because the real fault is (2): the identity swap of two like
   things that move by one vector.  Not it alone.
NEXT, in order: (a) the tracker — two like things that both move by one
vector each keep their name by their own displacement (the nearer
consistent place), to be found in the placement order at the 37c
tie-break with frames 155→156 as the test; (b) a divergence that names a
kind never asked triggers one directed press against it before any
replan; (c) 'went further than predicted' as a divergence kind, doubting
the push stop.
Referee: tw01 L1–L2 (65/49), tu93/bp35/ka59 L1, ls20 L2.  Snapshot s74.

### 37q (2026-10-17): LIKE THINGS NAMED JOINTLY; four more statements read off tw01's frames

Each found by the frames (FRAME_DUMP) and a fresh tracker replayed on them,
stated generally, A/B'd by switch, refereed on the chain:
1. LIKE THINGS KEEP THEIR NAMES BY THE LEAST MOTION, TOGETHER (scene):
   when things of one colour each read whole in several places and their
   first readings collide, the names are assigned jointly — one whole
   reading per thing, on cells no two share, least motion in all.  tw01
   frames 155→156: 4+4 against 12+4 — the swap of 37p is gone, and stays
   gone through 159.  (NO_JOINT_NAMES)
2. AN AGENT IS NO WALL OF ITS KIND (fiddle.blocking): a thing that
   travels on its own refused the walker where it stood and will stand
   elsewhere — a solid for this step, not a wall of its kind.  (BLOCK_AGENTS)
3. ...AND ALONG ONE AXIS (planner.Fixed.observe): a key whose every held
   translation runs along one axis does not send one thing diagonally; a
   displacement across that axis is two like things trading names.  It
   took 'ACTION3 moves the block by (-4, 8)' out of the carry.  (NO_AXIS_RULE)
4. A THING THAT GAVE WAY IS NOT THE GROUND (fiddle, ground learning): what
   stood on the cells the mover crossed and is not there after — moved,
   split off, vanished — was in the way; only what stays under the mover,
   or lies hidden beneath it, is ground.  And what shows where the mover
   STOOD is ground only where no thing now stands (the helper shoved a
   block into the walker's old place: frames 20→21 of the dump) — the
   mover's own body excepted (wa30's walker and its front row).  Before,
   the blocks' colour became ground and every plan walked through them.
   (NO_GAVE_WAY, NO_LEFT_TAKEN)
Each of these alone changed tw01's path; what won L2 at 49 on the old code
(the pair planned at once; the retract law's refusal) is replaced by a win
on the first attempt of each level: L1 28 (plan), L2 80 (plan).  L3 still
lost: step 13, the block pushed against the other block (a push into a
pushable — the chain push — unread).
wa30 L2 caught the first form of (4) (the front row read as a thing in
the walker's old place) and set its shape.
Referee (chain7, this code): 18/18 plain (tu93 L1 lost once to the first
form of statement 4 — a tile hidden under the walker all along is the
ground it walked on; only what ARRIVED in the mover's old place is a
thing — repaired, 28 / L2 36); climbs as before: wa30 L2, cd82 L2, m0r0
L2, vc33 L2, ka59 L2, ft09 6/6, ls20 L2, re86 L1, s5i5 L2; tw01 L1–L2
(28, 80 — first attempt each), tw02–tw04 9/9.  The chain script now lives
in tools/referee_chain.sh.  Snapshot s75.

### 37r (2026-10-18): THE SEALED PROBES, RE-MEASURED ON s75 (counts only)

Tushar: "did we move on sealed tasks?"  The sim holdouts were regenerated
(simgen, seed 7 — the names match the records) and the blind set 2 games
restored from the s31 repo zip (the answer files left out).
  SIM HOLDOUT, the core alone, L1: 6 of 27 — shf0 (new), shf2, shh2,
  shi1, shj0, shu2.  The last core-alone record was 7 (shf2, shh2, shi1,
  shj0, shu0, shu1, shu2): shu0 and shu1 are now lost, "died in 17",
  identically; shf0 is won.  Net -1.
  BLIND SET 2, L1: 1 of 6 (sx11, L1-2) — as at s68.
So: no.  Three weeks of statements earned on the public games and the
twist suite moved the public ladder (28 → 32) and the twist suite
(tw02–04 9/9, tw01 L1–2) and did not move the sealed sets, and cost two
holdouts (shu0/shu1, a death at the same action — the same cause, by the
counts).  The twist suite is a transfer test of the SAME mechanics; the
sealed sets are different mechanics, and nothing on them moved.
What the counts allow: the shu0/shu1 loss is a regression (they were won
by this core before) and may be bisected over the snapshots BY COUNT —
which snapshot first loses them — without reading their play; the
statement that broke them is then found on the public games.  Done next.
Bisected by count: shu0 is lost "in 17 (died)" on s52, s65, s68, s70 and
s75 alike — not this week's regression; the record of 7 was taken under
the 400-cap harness with the floor's turns, a different condition.  The
core-alone sealed count has stood at 6 of 27 since at least s52.

### 37s (2026-10-18): HYPOTHESES FROM EFFECTS, FIRST STATEMENT — a reading is oriented by what the keys move

Tushar: perception is the precondition; on unseen mechanics the wall is
the hypothesis generator, which reads goals off the still frame only.
vc33 L3 (frames dumped): columns raised and lowered by clicks on their
feet, a marker riding on each column's top, a mark of the marker's colour
drawn on the wall at the target height.  The reader proposed "same-colour
marker" with the WALL'S MARK as the mover (the smaller thing) and the
column's base as its marker; no key moves a wall's mark; seven readings,
no plan, twice.
1. A READING IS ORIENTED BY THE EFFECTS (core_play2.pursue): a symmetric
   reading — a thing to its same-colour marker, a thing onto the same
   shape — whose mover THING (not its colour: the mark and the base share
   one) answers no key while its marker thing does, is added the other way
   round: the marker is the mover, the thing its mark.  The mover may be a
   part (a mark drawn on a wall).  Added beside the original, never in its
   place.  (NO_FLIP)
   vc33 L3: "turned round: ... C15[3] -> align(M=15/3 -> C15[2] on the
   wall)" planned in 6 clicks and executed exactly — the first column
   level with its mark.  The level is still lost: the clicks couple the
   columns (raising one lowers another), and a click's law contradicted
   once at a column's limit drops the key from the successors — the next
   diagnosis there (a limit is not a contradiction).
Referee: the whole quick chain and climbs green (tu93 1-2, ka59 1-2, tw02
3/3, cd82 1-2, s5i5 1-2, ls20 1-2, wa30 1-2, m0r0 1-2, tw01 1-2, tw03,
tw04, and the plain L1s).
Sealed, counts only, on this code: sim holdout 6/27 (the same six), blind
set 2 1/6 (sx11 L1–2).  Unmoved.  Snapshot s76.

### 37t (2026-10-18): vc33 L3, two more statements; the level's coupling left

Tushar: leave the sealed sets; perfect the known games and build the
goal-from-effects mechanism; then read what the sealed class says.
2. WHAT THE PRESS ITSELF SENT IS NOT ON ITS OWN WAY (fiddle, agents): a
   thing whose law under THIS key — a click's as much as a key's — is the
   motion seen went as it was sent; it is no traveller.  vc33 L3's
   markers, raised by one foot and lowered by another, were declared
   agents after three presses and their laws thrown away; the planner's
   successors emptied.  (AGENT_IGNORE_LAW)
3. A KIND WHOSE THINGS ANSWER A CLICK EACH IN ITS OWN WAY IS ASKED THING BY
   THING (core_play2, after the readings find no plan): two clicked
   things of one kind with different laws make the kind's answer per
   thing; every unasked thing of it is clicked once (at most eight —
   hand-chosen), then the readings again.  vc33 L3: the four unasked feet
   gave the laws of the other columns; every marker now has a plan.
   (NO_ASK_EACH)
   Left: the columns are COUPLED — a foot raises its column and lowers
   another; a column whose partner stands at its limit refuses; the
   planner's per-thing laws carry the coupling but not the refusal.  The
   next diagnosis there is the refusal's condition (a conserved sum).
Referee green (quick chain, climbs, twists, ft09 6/6).  Snapshot s77.

### 37u (2026-10-19): ASK THE NEVER-ASKED; the lost known levels read off their frames, by gap

4. A KIND NEVER ASKED IS ASKED BEFORE THE LEVEL IS GIVEN UP (core_play2,
   after the readings and the per-thing asking): a kind on the frame
   that no law, refusal, push, click, walk or display has reached is
   walked onto (or clicked when nothing walks), at most four kinds of at
   most four things (hand-chosen); then the readings again.  (NO_ASK_UNASKED)
   wa30 L3 did not fire it — kind 12 there is a DISPLAY (it answered an
   arrival), and the report's "never asked" list was the cruder one.
THE LOST KNOWN LEVELS, READ OFF THEIR FRAMES (what each needs; the gap):
- vc33 L3: coupled columns (a foot raises one, lowers its partner; a
  partner at its limit refuses — a conserved sum).  Laws per thing carry
  the coupling; the planner needs the refusal's condition and a joint
  goal over the three markers.  PLANNING gap.
- wa30 L3: ACTION5 picks up / drops; a tile dropped in a region on the
  left appears in the pen on the right (the pen is a display of the
  drop region); the carry organ delivered two tiles and the block by
  accident of its walks, then planned the held tile straight through the
  wall.  ORGAN gap (carry + display as one mechanic).
- sb26 L1: four buttons, a click selects (an outline), ACTION5 confirms;
  a legend at the top gives the order; a bar counts the tries.  No
  reading at all (readings tried 0).  A SEQUENCE organ: a composite
  action (select, confirm) and an order read off a legend.  HYPOTHESIS
  gap, with a new action shape.
- sc25 L1: a snake-like figure (head, body, tail) turned and moved by
  the keys; two pattern boxes below.  Readings: markers and collects,
  none the mechanic.  HYPOTHESIS gap (a shape to make / a path to draw).
- g50t L1: a maze of corridors; ACTION1/2 shift the maze's kind by 6
  (the field scrolls or the piece slides); ACTION5 moves another thing.
  HYPOTHESIS + PERCEPTION (what moves: the piece or the map).
- dc22 L1: a 2-cell-step walker among constellations; the plan's second
  step refused with nothing in the way ("the mover stayed").  LAW gap
  (a refusal unexplained: the ground/thing that refused is unread).
- sk48 L1: many things move under one key (a platform game: things of
  six kinds translate by 6; extend/retract bars); the plan's first step
  refused by kind 6.  PERCEPTION + LAW (a body of many parts).
- sp80 L1: the walker dies on the second step of every plan ("stayed,
  nothing in the way", then died).  HAZARD gap (a death the ledger does
  not tie to a kind).
- bp35 L2, cn04 L2, re86 L2, tn36 L2, lf52 L2, lp85 L2, r11l L2, ar25
  L2, su15 L2, tr87 L2, tu93 L3, cd82 L3, m0r0 L3, ka59 L3: not yet
  re-read on this code; each to be dumped and classed the same way.
Order proposed: the LAW/HAZARD gaps first (dc22, sp80, sk48 — the
fiddle's statements, general), then the PLANNING gap (vc33 L3 — a
conserved sum), then the organs (sb26's sequence, wa30 L3's carry +
display), then the L2s.
Referee green on this code (quick chain, climbs, twists).  Snapshot s78.

### 37v (2026-10-21): the law and hazard gaps — dc22, sp80 (sk48 read, left)

5. A PART THE KEYS MOVE MOVES WITHIN ITS THING (planner, make_successors):
   a mover read as a part of a host — dc22's selector drawn on its 3x3
   panel — has the host's cells for its ground and nothing beyond; and a
   mover standing inside a box (a thing of another colour, small against
   the frame, surrounding it with room) has the box's inside.  dc22 had
   planned sixteen steps out of the panel; the second was refused "with
   nothing in the way".  Now: no plan for those readings (right — they
   are not the mechanic), the refusal explained.  (NO_BOX_BOUND)
6. A DEATH AS THE CLOCK RUNS OUT IS THE CLOCK'S, NOT THE PRESS'S
   (core_play2, at GAME_OVER): when the level's counter stood at its
   last cells before the press, no hazard is read off what the body
   touched and no press is marked fatal; the level is tried once more
   from the start within its budget (the retry the core gives a first
   look that spent the level).  sp80: the bar spends two cells a press,
   the level ends at zero; the core had marked the press fatal and
   "looked again for the killer".  (NO_CLOCK_DEATH)
   The mechanics behind them remain: dc22 is an assembly (a selector in
   a panel; clicking a shape on the right sets a part on the left) —
   an ORGAN; sp80 is a paddle under a budget with a ball that has not yet
   been seen to move — the next press to find is what launches it.
   sk48 read: a cart on a vertical track; one key shifts the cart, the
   track's segments and more by six (a scroll of part of the field); the
   reader took a track segment for the mover ("fits frame" of the cart's
   hollow).  Statement 1 wants a refinement there: among the things one
   key moves alike, the subject is what moves AGAINST the crowd, or what
   stays while the crowd moves (the camera's own).  To do.
   (5, narrowed by the referee: s5i5's marks ride their pieces — a part
   is bound to its host only when no key moves the host.)  Referee green
   (quick chain, climbs, twists, ft09 6/6, s5i5 L2 72).  Snapshot s79.

### 37w (2026-10-21): the remaining known levels classed; the size of the march

7. THE SUBJECT MOVES AGAINST THE CROWD (pursue ordering): when one key
   moves three or more things by one vector, a thing that key moves
   otherwise is the subject; its readings come first.  (NO_SUBJECT_ORDER)
   On sk48 it did not fire — the cart moves WITH its track (all by
   six); the cart is told from the track by something else (its parts,
   its place in the legend).  Left in: harmless, general.
THE L2/L3 LEVELS, CLASSED (frames dumped, reports read):
- ar25 L2 (mirror): the shape and its image move oppositely under the
  keys; "fits the same shape" has no plan.  ORGAN (the mirror law in
  the planner: the image follows the shape).
- su15 L2: clicks extend columns by one, ACTION7 retracts; a fill-the-
  bars puzzle; readings are markers of dots.  HYPOTHESIS (statement 2:
  a thing that grows under a key has its ends and marks as goals).
- tr87 L2: the table organ's second level (ACTION3/4 scroll by 42); no
  plan.  ORGAN.
- re86 L2: the selected piece refused by kind 13, held flat/ground; the
  ledger doubted both and the retry still found no plan.  LAW (what the
  selection may cross) — the ledger is doing its part.
- tn36 L2: "keys known []" — nothing moved anything the core pressed;
  the drawn legend of 37-era notes.  KNOWLEDGE (the asking: clicks on
  the legend's cells).
- lf52 L2: the leap organ's second level; explore 19, no plan 16.  ORGAN.
- bp35 L2: lost in one action at the quarter clock (the time sink of
  37n is spent on collect readings still); at 300 s it dies at 12.
  MECHANIC unread (clicks drop blocks).
- cd82 L3, m0r0 L3, ka59 L3 (dies at 100), tu93 L3 (dies at 4–8): not
  yet read beyond their reports.
THE SIZE: the lost known levels are some twenty distinct mechanics —
sequence (sb26), assembly (dc22), paddle (sp80), snake (sc25), maze-
scroll (g50t), track (sk48), coupled columns (vc33 L3), carry+display
(wa30 L3), mirror (ar25 L2), fill-bars (su15 L2), table (tr87 L2),
legend (tn36 L2), leap (lf52 L2), selection walls (re86 L2), drops
(bp35 L2), and the four L3s.  Each has so far cost a session to read
and one to three statements to win.  By the closing date that is not
all of them; the order stands as proposed (general statements first),
with the sealed reading after.

### 37x (2026-10-21): the progress loop — what counts, read from the presses (sb26 1–6)

THE CRUX RESTATED.  The core answered "what do I control" (laws) and
never "what counts".  A person opening a game watches for what changes
that they do NOT control and ties it to what they were doing.  That
record, generalised, is the goal.  agent/core/progress.py keeps it:
every change of a non-controlled thing on a press is a SCORE event
tied to a CAUSE (click on a kind, with or without a selection; arrival
on a kind; a key pressed over a kind).  Nothing in it names a game.
Built against sb26 and stated generally; the test of the loop itself
is whether the same loop, unchanged, moves other lost levels and the
sealed counts.

STATEMENTS (all in progress.py / progress_protocol in core_play2):
1. A SELECTION IS A MODE.  A click that leaves a mark (a thing that
   appeared or changed against the thing clicked, a part born on it)
   is a selection; the mark stands for the thing clicked until it is
   gone.  A click made while a selection stands is a click WITH it.
   The kinds whose click leaves a mark are the BUTTONS; the line where
   they stood when clicked is the TRAY, and the tray's other kinds are
   buttons too.
2. A FILL.  A click with a selection after which a thing of the
   selected kind lies ON the thing clicked, or AROUND it (a hollow tile
   holds the slot in its hole), is a fill: the slot took the selection,
   the selection is spent.  The slots are the like things of the
   filled slot's kind and size, hidden under a filling or not.
3. THE LEGEND.  The longest row (or column) of things of the buttons'
   kinds standing apart from the buttons and the slots — things, or
   PARTS of a backdrop (a terrain's other colours are things of their
   own; a part of several pieces is read piece by piece).  sb26 L1–4:
   the legend is drawn on a strip read as one terrain; L5+: the strip
   spans the frame, its colour is ground, the rings are things.
4. THE RUN'S TELLING (read_telling).  A press that returns many frames,
   paints things on the way and ends (nearly) as it began has TOLD
   something: the things it painted, in the order painted (a frame
   FILLED IN is the frame's: a thing whose hole holds the painted cells
   comes before a thing that merely covers them), and the place that
   blinked at the end (cells alternating over the animation's tail).
   Read for every press that returns six frames or more, kept in the
   record (progress.tellings).
5. THE RUN TELLS HOW FAR.  Each key is pressed once; a key whose press
   painted the legend's entries in order and blinked somewhere has run
   the slots and stopped: the slot it blinked at (the slot inside the
   blinking ring; else the ring's hole itself — the run stops at slots)
   takes the legend's next entry, and the key is pressed again.  This
   alone wins sb26 L1–L4: the second box (a call) needs no concept —
   the run goes where it goes and stops where the next entry is wanted.
6. A TILE IS USED ONCE (the supply).  A tile to select is taken from
   the tray first, then from a slot the run has not matched; a tile the
   run already matched is not taken — that is a SHORTAGE.  A shortage
   of a kind means the legend repeats a stretch, and a repeated stretch
   is a BOX's run (a box: a hollow thing whose hole holds slots; a CALL
   TILE: a free hollow tile of a box's colour).  When the earlier
   stretch already runs in a box, the call tile is placed where the run
   stopped; else the earlier stretch's first tile gives way to a call
   tile (the box takes the stretch; the entries from there are placed
   again as the run dictates).  sb26 L5 (6 e 8 8 e 8 8 b f with one e
   and two 8s): the e of entry 2 gives way to a ring-9, the box takes
   e 8 8, the second stretch is called.  L6 followed.
7. THE LEGEND GIVES THE ORDER, as a fallback: when no press tells and
   the legend has one entry per slot, the slots are filled in that
   order line by line and the keys pressed once more.
   Hand-chosen: ≤ 8 kinds asked in the mode probe; ≤ 2×slots+2 runs;
   a blink is ≥ 3 alternations over the last 24 frames; a box is a
   hollow thing of more than 24 cells; a tile ≤ 64 cells.
   sb26: L1 45, L2 51, L3 54, L4 53, L5 62, L6 62 — all by the loop,
   no sb26 statement anywhere in the code.
THE REFEREE (first run with the loop in):
- ls20 L1/L2 lost: the loop's "press again where it counted" spent the
  display's tokens before the display protocol ran.  The loop now runs
  AFTER the display protocol (an organ with a reading comes before the
  last resort).  ls20 L1 28, L2 47 again.
- s5i5 L2 and tw01 L2 lost: NOT the loop (lost with NO_PROGRESS and
  NO_TELLING too; s79 wins them) — statement 37w.7 (the subject moves
  against the crowd) took s5i5's PISTONS (extend/retract under the
  click that moves a column) for subjects.  Narrowed: "moved otherwise"
  is moved by another VECTOR; a thing that grows or shrinks under the
  key is the mechanism, not a subject.  s5i5 L2 72, tw01 L2 80 again.
  (37w's "harmless" was wrong; the referee was not run on it.)
- bp35 L1 lost once under two processes (timing; won alone).
- bp35 L1 lost its first attempt: the loop's "click every thing of kind
  14" (blocks dropped) ran before the exploration that finds the plan.
  The loop is now the LAST resort — after every reading, organ,
  exploration and fiddle; its presses are not free.  bp35 L1 28 again;
  sb26 1–6 unchanged (no mover: nothing to explore, the loop at once).
THE SEALED COUNTS (counts only; the loop before the move to the end):
  sim holdout 6/27 as before (shf0 and shi1 now by the loop, formerly
  plan and explore); BLIND SET 2: sx10 L1 WON by the loop — 2/6 games
  (sx10 L1; sx11 L1–2), from 1/6 since the set was sealed.  Nothing in
  the code names sx10; what the loop found there is not inspected.
FINAL REFEREE (the loop as last resort): quick list 18/18 first attempt;
  climbs as s79 plus sb26 6/6; twists unchanged; sealed counts rerun:
  sim holdout 6/27 (shf0 plan, shi1 explore again), blind set 2 sx10 L1
  by the loop, sx11 L1–2.  Public ladder 38/183.  Snapshot s80.
TRIED AND LEFT: dc22 L1 (a 10 000-line assembly; the record says "click
  kind 9 moves S23" sixteen times and nothing fills — an organ, not the
  loop's), tn36 L2 (the program organ's second level; the loop reads the
  panel's toggles as scores).  wa30 L3: the carried block moves on
  every press and floods the record (a carried thing is a controllable
  — a statement for the loop, to do).
NEXT: the loop on the remaining click puzzles with a feedback (su15 L2,
  vc33 L3); "what is carried is controlled" (wa30 L3); then the sealed
  classes (knowledge gap or perception gap) as agreed.

### 37y (2026-10-22): the loop's second game — su15 L2–L3 by merging up

Tushar: "grand idea is different and implementation is different" — the
code keeps improving under the same rules.  The loop's record gained one
relation and the protocol one plan; su15 L2 and L3 (never won) fell.
1. A CLICK CALLS LIKE THINGS TO IT; TWO THAT MEET BECOME THE NEXT KIND
   (progress.observe).  On a click with no selection: a thing that came
   to the click as itself (same kind and size; the NEAREST of its kind —
   one "coming" from farther while a nearer stood is the tracker's
   naming) teaches the REACH; a thing of a new kind standing at the
   click while two or more of one kind went (vanished, or one grew into
   it) is a MERGE of that kind into the new one.  A thing that vanished
   or changed kind under its own click was consumed, not selected (the
   mark rule narrowed).
2. MERGING UP (merge_up, in the protocol): every kind that has merged
   and every kind made by a merge is merged in turn — the lowest first,
   the nearest pair, the pair kept until it merges or refuses; within
   twice the reach the click goes between them, else one is called a
   reach toward the other; the reach is probed by doubling while calls
   answer and a call that goes unanswered BOUNDS it for good (a bound
   that is not kept let a misread reach of 12 send calls that cost
   penalties).  A click between two of a kind that does nothing refutes
   the kind.  Things in the band above the field are not merged
   (pictures).  Then the ladder's top is called once, well within the
   reach, to learn its law (the planner wants two observations), and the
   readings are run again.  ≤ 24 clicks; a box is a hollow thing of
   more than 24 cells (not used here).
3. THE LADDER CARRIES BY KIND (merges and reach in carry): when this
   level's readings want a kind not on the field that the carried
   merges make from kinds present, merging comes before the readings —
   su15 L4's eight dots; otherwise the junk readings walk the pieces
   onto one another and the clock goes.
4. THE TWIN'S PLACE (relations.py): the HUD twin's destination is a
   ring's hole; or, with no ring on the board, the one thing of a unique
   colour large enough to lie on (L2's disc, L1's ring drawn solid); as
   many such things as twins give one each (L3's two discs), a place
   another twin already lies on is taken; the HUD may hold other things
   of the twin's colour (a palette's swatch).  A twin's reading names
   its place (the reading key).
5. A THING THE CLICKS CALL LIES WHERE IT IS CALLED (pursue): a hole-less
   target of a called thing is no solid and reached = standing over it.
   For walkers the old rule stands (they stand against) — narrowed after
   the referee: wa30 L2 at the clock's last press.
6. A HOLE THE MOVER FILLS EXACTLY IS ITS OWN SHADOW, not a box drawn
   round it (make_successors: the block on its disc was bound by "the
   cursor's box"); a half-step is offered beside each call's step (a
   refused call is not the only call).
7. A THING MADE BY A MERGE IS A NEW THING TO THE LAWS: the tracker keeps
   the dot's name on the block; the block's positional click laws were
   the dot's (a 23-step junk plan); the merged product's laws by name are
   dropped.  (First form — any kind change — dropped the laws of wa30's
   walker wearing its facing colour: one press more, the level lost at
   the clock; narrowed to merges.)
8. THE CLOCK: a death with the clock at or under what one press can take
   (the rate, or the largest spend seen twice) is the clock's; the frame
   after a death is no witness (it may blank the clock — tu93 L2 was
   read as a clock death and lost).  A clock death leaves no safe prefix
   (the replay of what spent it is skipped).  A press that spends the
   clock is not free: the fiddle takes a quarter of what is left (≥ 4),
   and clicks outside the playfield (the band) come after the field's.
9. The click game's mover (look_again): the largest thing on the field
   of a kind with a relative law.  "Reach the odd thing" excludes things
   outside the playfield (first form — "another backdrop" by the colour
   round the thing — struck tu93 L2's targets; narrowed to the playfield).
su15: L1 14, L2 25, L3 39 (never won before; human baselines unknown
to me).  L4: eight dots, a 7-coloured hollow thing that eats a block
made beside it and the pictures turn colour — a new agent/hazard, left.
Referee: tu93 L2 and wa30 L2 regressions found and bisected (8, 9, 7);
final chain running at the time of writing.
   (Referee close: s5i5 L2 lost at 16 — the fiddle rationed by a clock
   that ends nothing (s5i5's bar drains and the level goes on); only a
   clock SEEN TO END A LEVEL in this game rations the fiddle now
   (carry clock_deadly).  tw01 L1's second attempt loses at s80 as well
   when run alone now (289/181 both): a flaky level of the twist suite,
   not this session's.  Quick list 19/19 incl. sb26; climbs as s80
   plus su15 L2–L3.)  Snapshot s81.

### 37z (2026-10-22): vc33 L3 — the coupled pumps planned jointly; the level's gate read

Tushar: "I keep improving the code regardless of the paper."  vc33 L3
(never won) was the surgical target.  Statements made, each general:
1. WHAT ONE GAINS ANOTHER LOSES (planner.Fixed.observe): under one key a
   thing of a kind that vanished by about the amount another of its
   kind extended has RETRACTED (law kept, min size 0), and a thing born
   by about the amount another retracted has EXTENDED — the tracker
   names an emptied column "gone" and a refilled one "new"; the law is
   of the column.  A born thing standing on a vanished thing's footprint
   inherits its name for the laws (`footprint`).
2. A THING THAT COMES BACK WHERE IT WENT IS THE SAME THING (scene.py
   ghosts): a vanished thing is kept 40 updates (hand-chosen); a thing
   of the same body appearing within its box (±1) and sharing an edge
   takes its name — vc33's columns at empty and refilled (sid 19
   regrown).
3. READINGS COUPLED BY A KEY ARE PLANNED TOGETHER (`_coupled_aligns`,
   `play_joint_mixed`): align readings are paired by kind (mover, its
   target by colour nearest the target cells); when one key moves
   things of two or more pairs the pairs are one plan: goal = every
   pair level on its axes (within 1.0), heuristic = the sum of the
   distances / 3 (one press moves 2 and may move two things), best
   first, 60000 states / depth 60, up to 8 replans (all hand-chosen).
   A press that moves nothing is remembered as a refusal with the
   thing's own footprint (`model.refuse(kind, r, c, v, size)`), a step
   the current state no longer allows is replanned without pressing.
   The mixed plan comes AFTER the singles (first before them — it took
   vc33 L2's one-pair level into a three-pair junk plan).
4. BEYOND A FLOOR IS FLOOR (make_successors `is_refused_`): a refusal
   of a thing along one axis refuses every place further that way on
   the same line.
5. A RETRACT LAW OF A THING THAT IS GONE AND LEFT NO LAST IS SPENT (the
   key is refused there; an emptied column cannot give).
Referee after these: quick list 18/18 (tu93 L1 WON again: the s79
baseline had it lost twice; s5i5 L1 50 as s80); climbs running at the
time of writing.
vc33 L3 still lost (50/52).  THE DIAGNOSIS, from the record and the
level's frames: the three readings are right (each float of a colour to
the mark of its colour, read as "turned round: the keys move this one"),
the seven keys' laws are right as far as they were asked (each key
moves water from one column to another: one float down 2, one up 2, or
a column retracts while another extends), the coupled plan is right and
finds nothing in 41–59k states because NO SEQUENCE OF PUMPING puts the
floats at their marks: a float rides its own column, and a mark is on
another column's wall.  The level has a second thing — a GATE between
two columns (kind 5 things clicked once in the fiddle, nothing
happened) that LIGHTS when the two columns beside it are level with it,
and a click on a lit gate EXCHANGES the two columns' riders.  The fiddle
clicked the gates unlit; nothing was registered, so nothing could be
read.  Class: KNOWLEDGE GAP of a new shape — a CONDITIONAL AFFORDANCE
(a thing acts only when a relation holds, and the relation is one the
keys can make).  What a reader would need, stated generally: (a) A
PRESS THAT RECOLOURS A THING WITHOUT MOVING IT HAS LIT IT, and a lit
thing is asked (clicked) before anything else — the fiddle's "ask what
just changed"; (b) A LIT THING'S EFFECT IS REGISTERED AS ITS OWN LAW
("click on lit G: the riders of A and B exchange"); (c) THE LIGHTING IS
A RELATION TO BE MADE (the planner pursues "A and B level with G" as a
sub-goal, then the exchange, then the levelling).  (a) is cheap and
general (lights, switches); (b) and (c) are the conditional law the
planner does not have.  The pumping alone cannot light a gate by chance
in a 21-press fiddle, so (a) without (c) will not win the level; left,
with the shape named.  Costs: ~25 presses of pumping along the chain
6→13→16→19→15 were what the plan wanted before the gate was understood
— the search was not the gap, the knowledge was.

### 37aa (2026-10-23): wa30 L3 — the handover; the referee's three bisections

wa30 L3 (never won) read off its frames: a hatched fence down the middle
the body cannot cross; the helper (kind 12) on the far side takes any free
block beside it and sets it in the pen; two blocks stand IN the fence at
the start and the helper takes them first — the level's own demonstration.
The carry organ had taken a block, found no way to its place, and left
it beside the fence, not in it.  Statements, each general:
1. WHAT WAS FOUND LODGED IN THE HATCHING PASSES THE HATCHING (carry, the
   first look): a thing of the carried kind standing with a see-through
   thing (under 4/5 of its box filled — hand-chosen) against it on two
   opposite sides was lodged in it; the held thing may pass that kind.
   By the THING (overlap_ok of the held core and its border with the
   hatching), not by kind: the body's nose wears the held thing's colour,
   and a kind's leave carried the body through the fence.
2. WHAT MUST BE HANDED OVER IS CARRIED IN FRONT: where the level has a
   strait (a hatching the things were lodged in; a carry that ended
   short of its place), the thing is taken from the side squarely behind
   it on the way to its place — the side whose way lies most along that
   way — and from no other while that side can be reached (else any);
   the place's own side need not be free there (the thing is handed
   over, not set down by us).  A block taken from the right was carried
   behind the body, and the body met the fence.
3. WHAT THE CARRIED THING PASSES, THE BODY DOES NOT (fiddle, planner): a
   thing wearing the carried kinds (the carry law's kinds, its core, the
   held colour) that moves with the body is carried — it teaches no
   flat and no passable for the body, and its kinds are no leave of the
   body's in the plan (the bond record may lapse with the tracker's cut;
   the kinds do not).
4. A BORDER IS OF THE THING'S OWN FORM: no larger than the carried
   thing's dims — the fence held the core's box and was bound to the
   body as its border, then moved with it.
wa30: L1 54, L2 72, L3 82 (never won before).  L4: a new level, lost
(no reading; four movers; left).
THE REFEREE'S THREE BISECTIONS (regressions of 37z's statements):
5. tw02 L1: ACTION4 was SPENT by a part (a piece of the wall the body's
   step covers) — spent only by a THING; tw02 L3: spent by the band's
   bar the key also shortens — not by the band's things; and finally
   not by the clock's bar by any name (the clock's sid, or the thing in
   its box of its colour).  "Seen to empty (min size 0)" was tried and
   lost vc33 L2 (its empty column was never seen with anything in it):
   a whole thing of the field, absent, spends the pump; a part or the
   band's does not.
6. m0r0 L2: BEYOND A FLOOR IS FLOOR extended a refusal by a WALL at the
   top of a column to every place below it (walls are terrain too).
   Withdrawn (switch FLOOR_EXTEND) until a floor can be told from a wall.
Referee close: quick list 18/18; climbs as s79 plus wa30 L3 and tu93
L1 (no regressions).  Public ladder 39/183.  Snapshot s82.

### 37ab (2026-10-23): the sealed reports classed by gap (counts only); the hatching carried by kind

THE SEALED SETS, RE-MEASURED ON s82 (the lanes; counts only, the combos
never inspected).  Sim holdout: 6/27 first levels won on the first
attempt (unchanged since s52), and 10/27 within the two attempts the
harness allows (four fall on "once more from the start, knowing the
keys" — the ledger's count has always been the first attempt's; Kaggle
scores the level however many actions it took, so both numbers are
kept).  Blind set 2: 2/6 (sx10 L1 by the loop, sx11 L1–2), unchanged.
THE REPORTS CLASSED, by a script that prints the class and nothing the
agent saw (no reading, kind, key or place):
- sim holdout, 17 lost: NO READING (hypothesis gap) on both attempts in
  12 of them; died (hazard or clock) in 3; one "a reading, no plan"
  (planning gap); one "plan diverged" (perception/model gap).
- blind set, 4 lost: PLAN DIVERGED on every attempt — a reading and a
  plan exist and the world leaves them (perception/model gap), never
  "no reading".
So the two sealed sets fail differently: the holdout's games are not
READ (the readings' vocabulary does not cover them — the gap the
abstraction document names), the blind games are read and MISMODELLED
(the laws or the things are wrong — the tracker's or the law-maker's
gap).  The next statement for each is a different organ.
THE HATCHING CARRIED BY KIND (carry_law, hatch_kinds): what the carried
things were found lodged in on one level passes on the next — wa30 L4
(seven blocks, the player fenced in a yard, three helpers outside, four
pens) now hands one block over and the helper sets it (1 in place);
the rest is the organ's approach plan inside the crowded yard (timed
out at 8k states, "beyond our reach") — left, with the shape named:
A YARD IS WORKED FROM INSIDE (the sides to take from are the yard's
free cells; the handover side is any fence side a helper can reach,
not the side toward the pen).
tn36 L2 looked at again: the left half is a DEMO (a program of four
six-segment glyphs with every segment lit, and two small pictures —
the board before and after) and the right half the player's board and
program; the legend is drawn, and reading it is the composition of the
legend organ with the program organ (as 37i said).  Not started.
Referee on the hatching carried by kind: quick list 18/18, climbs
identical to s82.  Snapshot s83.

### 37ac (2026-10-24): ONE SEALED GAME OPENED, AT RANDOM — shw0

Tushar: "whenever you think we are at the edge and hitting a wall, open
one game that was sealed at random and learn what the core was missing
… intelligence is also actively learning and memorizing the environment
and creating a causal framework which keeps updating on failures and
successes."  The rule changes from "sealed for ever" to "sealed until
the wall; then one, drawn blind, becomes a study game".  THE DRAW:
`random.seed(20261024); random.choice(<the 25 sealed games lost on the
first attempt in the 10-23 lanes>)` → **shw0** (sim holdout; lost on
both attempts, class "no reading").  From here shw0 is a STUDY game: it
leaves the sealed count (holdout 26 sealed; its line is reported apart),
and its combo, frames and report may be read.  Everything else stays
sealed.
WHAT THE CORE WAS MISSING IN shw0 (read off its record once opened):
the game is a maze with three goals to collect, and after EVERY press the
body also goes one cell east when the way is free — a standing push (the
combo's "drift_e").  The core's record had the reading right ("collect
the 3 like things", after the fiddle) and every law wrong: a key read as
(-3,+3) on one press and (-3,0) on the next (the drift blocked by a
wall) was CONTRADICTED by its second press, and with no law there was no
plan — 34 and 21 presses of nothing.  A person says it in one sentence:
"whatever I press, I also slide one to the right unless a wall stops
me."  Statements, each general, none naming the game:
1. THE DRIFT (planner.KeyModel): two readings of one key that differ by
   one step along one axis, the same step reconciling a second key's
   readings, are the key's OWN step and a drift the body takes after
   every press when its way is free.  The law kept is the own step (the
   axis-aligned reading); own, own+drift, the drift alone (own refused)
   and nothing (both refused) contradict nothing.  Under a known drift a
   reading is reduced to its own step before it is weighed; the slides
   the drift's steps looked like are dropped; the drift carries to the
   next level as a law of the game.
2. THE DRIFT IN THE PLAN (make_successors): after every move the body
   also goes the drift's one step when nothing stands there — a slide of
   one step; the cells aimed at are no obstacle to it (the goal the drift
   carries the body onto is reached); a known drift stands before the
   slide's votes.
3. COLLECT IN AN ORDER THAT LEAVES THE REST REACHABLE (the collect
   reading): where a one-way motion is known (a drift, a slide), the
   member from whose place the most of the rest still have a leg comes
   first, then the nearest — nearest-first stranded a goal to the west,
   where nothing can go.
shw0: L1 26 (human baseline 32), L2 12 (28) — the game whole.  Referee
running; the sealed lanes after it (shw0 reported apart).
Referee on the drift (after one narrowing: the drift is ONE STEP of the
body's own lattice — as long as the own step — and every voting key's
readings must fit it; the first form read wa30's nose into a drift and
L2 died at its 72nd press): quick list 18/18, climbs identical to s82.
Sealed lanes rerun without shw0: holdout 6/26 first-attempt, blind set
2/6 — the drift moved no other sealed game (its two siblings shw1–2
stay lost; what they are is not known and not looked at).  So the
lesson of shw0 is the drift's shape, held now by the law-maker for any
game with a standing push; the next wall's draw will say what else is
missing.  THE METHOD, stated: a sealed game opened at random is read
for the one sentence a person would say; the sentence becomes a
statement of the core; the referee checks it against the known games
and the remaining sealed games measure whether it transfers.
Snapshot s84.

### 37ad (2026-10-24): THE PARALLEL RECORD; tw01 L3 — the helper that turned

Tushar: "why is there no parallel tracker? the ally and the enemy point
… should have implemented that"; then: "a single input can have three
different resulting actions — identifying that is a basic
acknowledgement; identifying the pattern is intelligence: anything and
everything that happens, gauging why it happened and what would be
next."  The agents organ of 36w was a detector and a counted sign, not
a model.  Built now, earned on tw01 L3 (the twist he designed: the
helper pushes blocks to the pen on L1–2 and away on L3; lost on every
run since 37b):
1. THE UNIT OF A KIND (scene._learn_units/_split_units; _objs_by_units
   for the first look; goal_clusters keeps units apart): the smallest
   whole rectangle of one colour ever seen to MOVE alone is the kind's
   unit, carried across levels (`unit_form`); a thing that is a whole
   row or column of units is that many things.  tw01's pair of blocks
   was one 32-cell thing to every reading and every law — a push that
   moved half of it was read as a retract and an extend (the wrong law
   that worked on L1, 37o), and the plan was lost in the halves.  L2
   went from 74 presses to 44.  (The split-by-move read as a push, not
   a pump, was tried and withdrawn — switch SPLIT_LAW: without the
   wrong law L1 loops on the pair; the unit makes the question moot.)
2. WHAT STANDS WHERE THE PLAN PUT THE MOVER IS THE MOVER (the
   executor): a thing of the mover's kind and size at the predicted
   place means the step held under whichever name — two like things
   had traded names between the plan and the press, and "not where
   predicted / replanning" ran fourteen times over on the same press.
3. THE PARALLEL RECORD (agent/core/agents.py, fed from the fiddle every
   press; printed in the failure report): per agent its STEPS and
   stride; its WANT — the kind it keeps closing on (three approaches
   and more approaches than retreats, hand-chosen); its ACTS on
   arriving beside a thing of its want (took / carried / pushed /
   nothing); its SIGN against the goal's place — the cells the pursued
   reading aims at — by whether what it moved went toward them (ally)
   or away (foe); and a PREDICTION of its next step (one stride toward
   the nearest want-thing, the larger axis first, the free way) checked
   by the next frame, three misses in a row halving the want's record.
   Read off tw01 L3 by the record alone: "kind 6: stride 4, goes toward
   the nearest of kind 14 and on arriving pushes; foe".  A foe is not
   stepped aside for (the wa30 L2 rule) and not waited on (the carry's
   "in the helper's hands").
tw01: L1 54, L2 44, L3 10 (human baseline 8) — the twist suite whole
(tw01–tw04, 12/12) for the first time.
Not yet: the planner moving agents in its search (an ally's reach for
wa30 L4's handover, a foe's path as a moving obstacle); the outcomes of
our own keys conditioned beyond the drift (the blind set's "a plan the
world left").  Referee running at the time of writing.
Referee (two narrowings on the way): the unit marked every lone unit a
thing apart — two crates' cores in wa30's pen, and L2's reading order
moved one press at the zero-margin clock → only pieces split from one
component are kept apart; the unit read s5i5's pistons as rows of
units (a bar growing by one unit-length looks like a unit joining) →
a colour whose things STRETCH is elastic and has no units, and a growth
or shrink by exactly a like thing joining or parting (a vanish, a
birth, a covering of that amount, that press) is no stretch.  Then:
quick list 18/18; climbs identical to s82 plus tw01 L2 44 (was 74) and
L3 won.  Twist suite 12/12.  Snapshot s85; sealed lanes next.
Sealed lanes on s85 (counts only; shw0 apart): holdout 6/26 first
attempt, blind set 2/6 — unchanged.  The record is kept on every game
with an agent now; what it has yet to do is be ASKED by the planner.

### 37ae (2026-10-24): THE FIFTH TWIST — making sense of the feedback (tw05, the jump)

Tushar: "the program needs to make sense of the outcomes / feedback and
interaction — when I press 'up' the game can send me 5 pixels instead
of one; it could be a Super Mario Bros-like 2D game."  A fifth twist
game was built for it (tools/platgen.py → tw05): a side-view room with
ground, ledges, a pen; left/right walk a cell; up is a JUMP whose arc
is shown frame by frame; a FALL of one cell a frame until something
holds the body; pits with lava (a death, the level begins again).  The
physics is the game's, not the level's: a RUNNING jump — up after a
step in the facing direction — goes five cells instead of three on
every level; L3's pit needs it.  Every level is proved solvable by a
search over the engine; the baselines are 3 / 4 / 6.  Earned on it, in
order, each by a diagnosis on a known level:
1. THE PATH WITHIN A PRESS (fiddle.observe_path): the body's place in
   each frame of the press is its path; the first frame before the
   press is part of it.  What the press moved the body by is the
   path's steps BEFORE the trailing fall (`path_own`), not the net —
   the hop off a ledge was read as "down 28".
2. THE FALL IS THE GROUND'S LAW: a trailing run of equal axis-aligned
   steps that differ from the first is a fall; its way, seen twice (a
   fall of three frames or more counts twice), is GRAVITY — the slide
   of every key, standing before the votes (model.slide() returns it
   first), carried across levels.  WHAT THE BODY RESTS ON HOLDS IT: the
   colours just beyond the body the way of the fall in the press's last
   frame are a SUPPORT; the plan's slide stops at them (the ground is
   terrain to the tracker and refused no press — it had to be learned
   as what the body comes to rest on; the plan fell through it before).
3. THE PLANNER UNDER GRAVITY: the active body defaults to the mover
   (gravity on nothing moved nothing); the reading's axes include the
   gravity axis (a plan accepted a pit under the pen); the `_no_slide`
   fallback — "theory on one sight" — is not tried when gravity is
   known (it planned into the pit).
4. THE OUTCOME CONDITIONED ON WHAT CAME BEFORE (KeyModel.cond,
   cond_law, has_cond; make_successors' `_prev`/`_head` state; carried
   by kind): every reading of a key on a thing is kept with its
   CONDITION — the press before it and the thing's heading.  When a
   key's readings of one thing differ and neither a drift nor a slide
   explains them, the coarsest condition under which each value saw
   one outcome is the law — by the press before, by the heading, or by
   both; a condition never seen is not planned with (a press there will
   tell).  The record on tw05: "ACTION1 after nothing → 12; after
   ACTION4 → 20", read on L2 without a death (the plan's third press
   landed two cells past its prediction) and carried into L3, where the
   first plan was the baseline route — jump, step back, step forward,
   the running jump across the pit, walk, walk — and every prediction
   held.
5. AN ARC IS NO SLIDE (KeyModel.arcs, set from the path): a press whose
   path bent was read frame by frame; the slide rule ("same way, not
   the same distance") and the flicker rule ("a step far longer than
   the key's is two things confused") do not touch it.
Hand-chosen: two distinct outcomes before a key is conditioned; a fall
is two or more equal steps; gravity at two votes.  Switches: NO_COND,
NO_GRAVITY.
tw05: L1 6, L2 4, L3 6 (baselines 3 / 4 / 6).  Referee running at the
time of writing.
Not yet: the parallel record consulted by the planner; the sealed
set's "a plan the world left" re-read against the conditioned outcome
(a wall-at-the-time reading is a refusal, not a condition — refusals
are kept out of the conditioning on purpose).
Referee, first narrowing: ka59 L1 died on its first attempt (was 91) —
a part's half-step against a wall under ACTION4 (6, then 3) was
"conditioned" on its heading, by coincidence, and the plan built on
it.  Readings all the SAME WAY that differ in length are one motion
stopped short (a wall, a slide, a misreading) and are not conditioned
— unless the path was read frame by frame and bent (the arc: the
jump's reach).  Readings in DIFFERENT ways remain conditionable (a
"forward" that goes the way of the heading).  ka59 L1 91 again.
Second narrowing, from tw05 itself under the referee (L3 in 9, a
death in the lava first): the arc flag was per press and not carried,
so at L3's start — no press yet — the same-way readings 12 and 20 were
"one motion stopped short", the conditioning lapsed, and the plan was
made without momentum; a reading "into the odd thing" (the lava, never
seen to kill) was pursued instead.  An arc, once seen, is the key's way
with the thing (a press against a wall shows no path and unsays
nothing), and it carries by kind like a slide.  tw05 6 / 4 / 6 on three
runs.  Referee rerun on the final code.
Referee on the final code: quick list 18/18, identical to s85; climbs
identical to s85 but wa30 L2 70 (was 72 — no conditioning and no arc
fired on wa30: the two presses come from the gravity segment's planner
defaults, the active body as the mover or the slide axis among the
reading's axes, not from anything this entry claims) and tw05 6 / 4 / 6.
Twist suite 15/15.  Snapshot s86; sealed lanes running (counts only).
Not built: the contrast experiment (each key pressed after each other
key when no plan exists) — tw05 taught the condition through the plan's
own misprediction, and no level yet asks for the experiment.
Sealed lanes on s86 (counts only; shw0 apart): holdout 6/26 first
attempt, blind set 2/6 games — unchanged.  The conditioned outcome
moved nothing sealed: the blind set's "a plan the world left" is not a
condition of the press before (or it is one the plan never meets).

### 37af (2026-10-25): wa30 L4 — the crowded yard, statement by statement

wa30 L4 (seven blocks, the body fenced in a yard of six, three helpers
outside, five pens): "no reading; four movers; left" (37aa), then "one
block handed over, the approach plan timed out at 8k states" (37ab).
Taken as the next known wall, by diagnosis on it, in order (each run
is eight minutes):
1. The conditioning of 37ae widened: every outcome but a refusal (a
   turn, a growth) can be conditioned; the FACING is the last key that
   moved or turned the thing (a refused press turns nothing); an
   outcome seen once is no evidence (two sightings over all records);
   the records of a key read two ways carry by kind whether or not
   the condition is read yet (the sightings add up across levels —
   tw05's jump, 12 then 20 once a level, was contradicted on each
   level and never carried), and a conditioned kind without a law
   here takes its commonest outcome as a stand-in; a condition never
   seen whole is answered by the coarser conditions it shares with the
   record.  Said honestly: wa30's body does NOT answer to (press
   before, facing) — the same pair saw a turn and a move — and the
   organ, applied honestly, reads no condition there.  What it needed
   was simpler:
2. A TURN IN PLACE IS NO MOVE (KeyModel): a key seen to turn a thing
   where it stands (its centre within its own extent) and, another
   time, to move it, MOVES it — the turn is the press that faces the
   thing, a refusal that changes the picture, not a second law; and
   the thing that turned to the key answered it — the law is no
   provisional record any more.  (The carried 'left' was replaced by
   the first press's turn and the move that followed "contradicted"
   the turn; no plan could go left.)
3. EVERY THING OF THE TARGET KIND IS A PLACE (carry): a level may have
   several pens, each holding as many as fit — five pens, seven slots,
   where the carry knew one pen's two.
4. A REFUSAL WITH SOMETHING IN THE WAY SAYS NOTHING OF THE LAW
   (fiddle → KeyModel.observe(blocked=)): a provisional law is dropped
   after two refusals only when the cells its step would take were
   free — the body's 'left', held over three levels, was dropped by two
   presses against blocks.
5. A CONDITIONED LAW COUNTS ONLY ON A THING THE KEY HAS A LAW ON
   (has_cond): the helpers' readings under our keys are records, not
   laws; they had put the press-before into every plan state and the
   approach search timed out.
6. A WALK WITH NOTHING MOVED (carry): when the way to the thing, shoves
   and all, is beyond the search, the way with every other pushable
   thing frozen is sought — a far smaller world.  Second, not first: an
   equal-length walk found another way shifted L2's zero-margin clock
   by the helper's timing, and L2 died at 72.
7. TWO WEAR THE MARK (selected_now): ours is the one the keys move or
   the one against a body the keys move; a marked thing nobody of ours
   touches is in another's hands — the block the helper carried off
   stayed "selected", and every law was unknown under that selection.
8. A BOND IS A TOUCH (carry): a thing bound to the body that no longer
   stands against it is let go of — every walk had been planned with a
   block the helper had taken.
L4 now: 3 of 7 in place within the attempt's 100 presses (was 1); the
remaining approaches still time out — the next diagnosis is the
helper-held block's "law" under our keys (S37 moving under ACTION5 in
the plan).  Referee running on 1–8.
Referee on 1–8: quick list 18/18; climbs identical but tw01 L2 49 (was
44) and L3 34 (was 10) — a block the helper pushed left under our
'right' was a reading of 'right' conditioned on the press before, by
coincidence, and the plan carried that condition in its state (the
search's tie-breaks shifted).  What is popped from the laws as an
agent's own step, or a thing in an agent's hands, is now struck from
the conditioned records too (KeyModel.unrecord, this press's records).
tw01 54 / 44 / 10 again.  Referee rerun.
Referee on the final code: quick list 18/18; climbs identical to s86
but tw05 L2 in 5 (was 4: the second sighting is now L2's own press).
Snapshot s87.

### 37ag (2026-10-25): wa30 L4, second round — the fence read whole; the clock read per tick

Continued on L4 by diagnosis, each a general statement:
9. UNDER A SELECTION, WHAT VARIES IS THE SELECTION'S (KeyModel.moving):
   a thing whose law under the key differs between the selections seen
   is unknown under a new one; a thing whose law is the same under
   every selection keeps it (the body walks the same whatever is held
   — under a block never held before every key was unknown and the
   body could not move in the plan).  And a refusal with something in
   the way under a selection teaches nothing of the thing there
   (blocked_none): its law over all stands.
10. AN AGENT'S STEP IS STRUCK UNDER EVERY SELECTION (KeyModel.forget):
   the selection's records kept a helper's step as a law of our key,
   and the plan moved the helper.
11. A MATERIAL IS READ BEFORE THE THINGS ON IT (entities.read_frame):
   same-colour pieces within a cell of one another are tried as one
   thing first; when their union repeats — a strip, or a RING of strips
   (each side's strip periodic; _periodic) — they are one material, and
   no piece of another colour standing against them is glued to them.
   wa30 L4's hatched fence was read in six pieces, two with blocks
   glued in, and was a solid fill of its colour (flat, pushable) to
   every organ.
12. A MATERIAL REFUSES WHAT WAS NEVER SEEN ON IT (make_successors): a
   textured thing is a wall to a mover whose kind no sight put within
   one, and lets through what was (passable, the carry's hatching
   record, any overlap the world has shown).  A texture is never
   shoved, and not flat for a mover never seen within one.  ON the
   material means beside it on two sides at least (read_materials): a
   ring has everything in its box, and the body in the yard "stood on"
   the fence.
13. WHAT WON THE LAST LEVEL IS TRIED FIRST: a level won by the carry
   opens with the carry, before the paint protocol's region test (ten
   clicks on the pens).  WHAT THE LAST LEVELS LEFT SILENT STAYS SILENT:
   a kind clicked with nothing changed is not clicked again to see, on
   this level or the next (silent_click_kinds, carried).
14. A THING THAT MOVED WITHOUT US IS ON ITS WAY (carry): one that stands
   elsewhere than at our last look, though we never held it, is in
   another's hands or on its own way, and not ours to take this round
   — the body had chased a block the helper carried along, a press
   behind it each time; wa30 L2 went from 70 presses to 60.
The clock, measured on the engine: the bar at the foot shrinks by one
cell every TWO TICKS; every key is a tick (ACTION5 too), a click is
not — and a click ticks nothing (the helpers stand).  Waiting on the
helper costs what walking costs.
L4 now: every block of ours handed over by press ~70 (was: approaches
timing out, refusals at the fence one place at a time); 4 of 7 in
place when the clock runs out — the helpers' pace is the wall.  What
is left is the ally's reach: which fence side to hand each block over
on so the three helpers work at once and travel least — the planner
asking the parallel record.  Referee running on 9–14 (with 1–8).
Referee, first narrowing: s5i5 L1 lost in 13 — "one silent click speaks
for its kind" deleted the fiddle's own repeated clicks on the pistons
(a piston at its limit is silent for where it stands, and a thing
clicked again on purpose may answer the second time).  Now: a kind is
silent after TWO silent clicks, and only the cheap probe of the clicks
(the keys known, the clicks not) skips the rest of the kind; a
fiddle's own sequence is never cut.  s5i5 L1 50 again.  Referee rerun.
Second narrowing, m0r0 L2 (was won in 44; lost, died ×2): the
material-first reading made its checker strips one material instead of
the lone cells the body had been GLUED to on touching them — the old
reading's "body vanished into the checker" was the death the plan
lived by; honestly read, the body stands beside the checker and is put
back, and the hazard organ then has to carry the whole load: a texture
of a hazard colour is a hazard (kind colour+16, _with_material_), a
hazard entered is not a wall in the plan (solids), a material that
kills is entered, not refused.  Those three are kept — but the checker
still won the day: the put-back press's reading of the keys (tried as
"a put-back teaches the key nothing", PUT_BACK_SKIP) is part of what
m0r0 lives by in a way not yet understood (L1 41 → 44, L2 lost with it)
— WITHDRAWN to a switch; and the material-first reading itself (right
about wa30's fence, L4 still lost with or without it) is WITHDRAWN to a
switch (MATERIAL_FIRST=1) until the hazard organ reads materials.  The
ring-of-texture test stays (harmless without the pre-pass; the fence's
pieces are not merged without it).  m0r0 41 / 44 again.  Referee rerun.
Third narrowing, re86 L1 28 → 30: "the same under every selection
seen" (statement 9) had one selection's word for it; now two
selections seen at least, else what differs makes the key unknown
under a new selection, as before.  re86 L1 28.
Referee on the final code: quick list 18/18, identical to s85; climbs
identical in every won/lost but wa30 L2 61 (was 70) and L3 80 (was
82); the lost levels' press counts vary as ever.  Snapshot s88.
Hand-chosen this session: two selections seen; two sightings per
outcome; two silent clicks; the walk-only search's 6000 states; the
ring band of eight or fewer.  Switches added: NO_COND, NO_TURN_RULE,
OLD_CTX_RULE, NO_WALL_PATCH, TEXTURE_FLAT, OLD_MATERIAL, NO_CARRY_FIRST,
NO_SILENT_KINDS, NO_RING_TEXTURE, HAZARD_SOLID, MATERIAL_FIRST (off),
PUT_BACK_SKIP (off), COND_DEBUG, CTX_DEBUG, PRESS_DEBUG, SKIP_DEBUG.

### 37ah (2026-10-25): THE "NO READING" CLASS, READ FROM HUMAN REPLAYS — three loop statements; two stories corrected
Direction set by Tushar: first the class the sealed ledger calls "no
reading" (12 of 17 lost holdout games, both attempts), then the agents
in the planner (the ally's reach).  Working mode set the same day:
write what the game IS before any code; he corrects the story; only
then statements.  The sealed games cannot be looked at; the public
games the core loses the same way can, and arcprize.org keeps a full
human replay of each (every press, every frame, read as JSONL from
/api/recordings).  Read as traces: g50t, sp80, dc22, sk48, sc25 (first
levels the core cannot start) and tn36, lf52, bp35, lp85 (second
levels it cannot read).  The note is claude/no-reading-class.md.
What the traces agree on: the person's first presses are the same few
experiments (two or three of each key, one click per distinct thing),
then the person plays; and every "no reading" first level in OUR
ledger already had a reading in its report — what failed was the
model of where the mover may BE ("the mover stayed where the plan
moved it; nothing in the way"), after which the law was dropped on
re-test, which is what "no reading" looks like from inside the ledger
on the second attempt.  Three loop statements, each coded generically:
 1. THE KEYS OF THE GAME ARE WHAT THE GAME OFFERS — the engine's
    available_actions; nothing else is pressed (wa30 clicked ten times
    in a game with no click).  step() refuses a key not offered
    (PRESS_ANY=1 to press anyway).
 2. THE MOVER WALKS ON WHAT IT WAS SEEN ON — the ground a mover
    uncovers, when it is not the background, is its way and the
    background is a bank until a press shows otherwise (fiddle
    .uncovered, model.track_way; the plan's walls take the background
    when a track is known; NO_TRACK_WAY=1).  g50t's ring now stays on
    its track (no "stayed" divergence).
 3. A REFUSAL WITH NOTHING IN THE WAY IS A BOUND, NOT A LOST LAW — two
    background-only refusals at the same coordinate from different
    places make a line the mover does not cross for that key
    (bound_lines_ in is_refused_; NO_BOUNDS=1); the law stands.  sp80's
    bar no longer plans above its range.  The edge rule now respects
    the track walls (wall0_), found when it removed them.
Referee on 1–3: quick list 18/18, same as s85 (ls20 28 → 25: the
unoffered presses gone); climbs identical in every won/lost to s88,
wa30 L2 59 (61) and L3 78 (80), ls20 L2 45 (47); re86 L2 still lost
(three attempts now, one at the cap).  Snapshot s89.
THE CORRECTION.  Re-reading g50t and sp80 PRESS BY PRESS (what each
press changed, not the first frame and the key list) showed the first
stories wrong, and the class misnamed twice over.  g50t L1: the 8
things are a PAD at a branch end, a WIRE, and a GATE across the track
that slides off the track only WHILE something stands on the pad (a
lever); ACTION5 is a REWIND — the ring returns to its start and two
presses later a colour-2 twin re-walks the ring's record one recorded
press per press, refusals included; the person walked to the pad,
rewound, walked to the gate, waited four presses for the twin to reach
the pad, walked through, home: 51 presses, three rewinds.  No bridge,
no gap, nothing carried.  sp80 L1: a TAP, a BAR, two CUPS; ACTION5 is
a RUN — water leaves the spout, runs straight, spreads along the bar
it hits, falls off both ends, fills the cups it enters; all cups
filled ends the level, else the board is as it was and the clock has
ticked (the person used runs as tests on L4–L6).  The core already
reads the run as a telling (progress.read_telling: painted the bar,
blinked one cup, restored) and makes nothing of it.  So the class is
two things: a mover's-domain fault (statements 2–3, now fixed) and
organs the core has no word for.  Three more statements written, not
coded, for Tushar to correct first: (6) a run's telling is a goal and
a test (kind blinked = kind to fill, count = score, thing painted =
mover; places tried nearest first, one run each); (7) a thing that
moves when the mover stands somewhere and back when it leaves is a
lever; (8) a rewind key makes the record a second body.  Statements
4 (a box the mover cannot leave is a menu — dc22) and 5 (a row of
distinct colours apart from the board is a legend of order — sk48)
stand as written.
Hand-chosen this session: the bound needs two refusals from different
places; a track needs one uncovering.  Switches added: PRESS_ANY,
NO_TRACK_WAY, NO_BOUNDS, TRACK_DEBUG, GROUND_DEBUG, DEATH_DEBUG.

### 37ai (2026-10-27, written up 11-01): THREE ORGANS FROM THE HUMAN REPLAYS — the run, the lever, the twin; s90
Tushar: "work on improving stuff; we are closing the competition; learn
from human cues in all the blockers."  Each blocker read press by press
from its arcprize.org replay (what each press changed), the story
written first (claude/no-reading-class.md, corrected twice), then one
organ each, stated generally.
THE RUN (sp80; statement 6; agent/core/runkey.py, run_protocol): a key
whose press draws a TRACE of one colour from a SOURCE over the board
and takes it back is a run.  Its telling: the things that FLASH at the
end are what the run is about (the cups that the water missed, the
floor it spilled on) — the goal is a run at which nothing flashes; the
things it ENTERED are filled; the thing it PAINTED is in its way.  The
trace is a LAW the plan runs itself (runkey.flow): the ink leaves the
source with the source's breadth, runs straight over the background,
spreads along the near face of what it meets, both ways, as far as the
thing lies beneath, falls where the face is open beyond with the same
breadth, stops where the face is shut, and is lost in the ground (what
is read as terrain).  The law is trusted only when it redraws every
trace seen (Jaccard ≥ 0.8, each run against its own frame); then the
movers (what the keys translate, and the selectable kinds) are placed,
one at a time, where the law says the fewest flash — the shortest
placement first, a partial gain of more than six presses not taken, a
move that did not land as the law said ends the round; the run key is
pressed once at zero.  When the law does not hold, the run is a TEST:
one run per place on the trace's line, nearest first.  The run protocol
goes before the paint protocol (which read the same press as a pour and
walked fifteen places), and FIRST on a level after one it won (the run
key carried: one press reads the new level's telling before any
reading).  sp80 L1 won in 10 (second attempt — the first dies fiddling
on the 32-press clock), L2 in 24–25 on the first attempt (human 11/18).
L3 (three taps, four bars, bars that pass through each other) still
lost: the tracker loses the bars' names when they overlap.
TWO READINGS UNDER IT.  (a) TWO THINGS THAT EXCHANGED PLACES EXCHANGED
COLOURS (scene.py): A's exact cells wear B's colours and B's wear A's,
the same shape — nothing passed through anything; both recoloured in
place.  ONLY UNDER A CLICK on one of them: a colour that passes to the
thing pointed at is a choice; a colour that passes to the next thing on
a key is a step (tu93's walker hops node to node and its old node shows
through — the frames alone cannot tell the two apart; the pointer can).
Not when a thing's old cells show a thing known to lie under it.
(b) A COLOUR A THING PUTS ON AND OFF IS A LIVERY, NOT A KIND: a thing
recoloured whole where it stands makes both colours elastic — no unit
(sp80 L3: the chosen coat's unit, learned on a 3-unit bar, split the
6-unit bar in two and every press read as a birth and a death).  And
the fiddle follows a click that PASSED A COLOUR between two things with
one key press (did the click hand the keys to it?) — only an exchange,
never a toggle (ka59's marks).
THE LEVER AND THE TWIN (g50t; statements 7–8; agent/core/lever.py,
twin_probe, twin_protocol).  A lever: a thing that moves when the body
stands on another and back when it leaves — two sightings (entered X, Y
moved by v; left X, Y moved by -v); Y may be a PART of a compound thing
(the gate on its wire), read by the frames' colours where the tracker
reads the slide as growth; the pad of a compound is only the cells the
body stood on.  A rewind: a key that puts the body back at the level's
start from elsewhere, NEVER having stepped it (ka59's hazard sends the
body home under an arrow: a put-back), from TWO DIFFERENT PLACES or by a
jump no step makes (bp35's arrow walks the body back one step), never
on a press that scrolled the screen.  The twin: a thing of the body's
shape standing where the record puts it, by shape and place not by
name (the tracker gives the new twin the old one's name) — and the
frames taught what the human's trace had hidden: the twin walks ONE
STEP PER ACCEPTED MOVE of the body, not per press; it appears only for
records of two moves or more; the next rewind replaces it.  The
protocol: a twin suspected but unseen — one rewind on purpose after
two moves, then two moves that are not the record's first (the twin
walks the record; the body must not stand on it); the lever unknown —
stand once on each place in reach whose cells overlap a thing (nothing
solid but the twin: a refusal costs one press, and the gate's kind is
the pad's); then: the goal that has a way with the gate drawn open
(the readings' targets in turn, the band's marks have none), one
rewind to clean the record (the twin would otherwise walk the
fiddle's wanderings through the corridor where the body waits), the
shortest way onto the pad, the rewind, the way to the goal, waiting at
the gate — a step back and forth, counted in moves — until the twin
has walked the record and holds the pad.  The rewind key is an undo
key to every plan (pressed on purpose only); the fiddle stops when a
twin is read.  g50t L1 won in 44–53 on the FIRST attempt (human 51).
Narrowed on the way: bp35 L1 (the run must be taken back; the arrow
back is no rewind), tu93 L1 (the exchange only under a click), ka59 L2
(the stepped key; the select probe only after an exchange).
Referee: quick 18/18 as s85 (ls20 25); climbs identical in every
won/lost to s89 (40 wins).  Sealed lane a unchanged on its first 12
(counts only); the container was reclaimed before b and c.  Snapshot
s90.  THE BUNDLE (11-01): tools/build_core_submission.py listed neither
progress, journal, claims, agents nor reader — nor the two new modules;
added, in import order; checked through the adapter offline (sp80
reaches L3).  Delivered as my_agent_core_v38.py.
dc22 and sk48 read press by press (the note): dc22 is a cursor that
walks coloured cells, whose pictures are switches moving their like-
coloured pieces between two places (not a palette); sk48 a crane with
an arm that extends, retracts and pulls what its end touches, ordered
by the legend.  Neither coded.
Hand-chosen this session: Jaccard 0.8 for the law; six presses for a
partial gain; the test's six places; two sightings for a lever; two
origins for a rewind; two matched steps for a twin; three stands;
twelve waiting pairs; two probes.  Switches: NO_RUNKEY, NO_LEVER,
NO_SWAP_RULE, NO_LIVERY_ELASTIC, NO_SELECT_PROBE, SWAP_DEBUG,
LEVER_DEBUG, TWIN_DEBUG, PLANB_DEBUG, SPLIT_DEBUG.
