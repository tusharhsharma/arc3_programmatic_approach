# Concept-Composition Simulator — design note (2026-08-29, Tushar's proposal; measurement section 2026-09-03)

## The idea (approved direction)
Learn the concept inventory from ARC-2/ARC-3; generate SYNTHETIC MINI-GAMES
as permutations and compositions of those concepts (not full game clones);
run the agent; categorize failures by WHICH CONCEPT was mis-hypothesized;
every fix earns a permanent confirmation suite. Iterate toward measured
generalized learning.

## Why it is sound
- Precedent inside this project: the mz01-05 hazard fixtures were exactly
  this at small scale, and drove most robustness gains.
- Attacks the measured weakness: naked-scientist cold-start = 2 levels;
  hidden games are novel COMPOSITIONS of partially-known concepts.
- Matches the benchmark's theory: fluid intelligence = skill-acquisition
  efficiency on novel tasks given fixed priors. The sim provides a
  DISTRIBUTION of novel tasks to measure actions-to-first-clear on
  held-out compositions — the closest local proxy to the Kaggle score.
- Paper Prize asset: a concept-composition benchmark + generalization-gap
  measurements is a publishable contribution.

## Failure modes to design against
1. GENERATOR LEAKAGE: recombining only what we implemented proves nothing.
   Defense: compositional holdout (train pairs, test unseen triples) +
   mutation operators (inverted win conditions, out-of-range parameters,
   concepts wearing other concepts' visuals).
2. GOODHART: the sim is curriculum + diagnostic, never the optimization
   target. DEV = validation, Kaggle = referendum.
3. LIMIT: the sim cannot reveal a MISSING core prior — only real-game
   failures do. The DEV-zero diagnosis loop stays; the two loops feed
   each other (real failure names a concept -> generator multiplies it).
4. (2026-09-03) THE BATTERY BLINDNESS: a DEV battery whose gates were tuned
   on the very games it tests cannot see off-distribution behavior. Three
   identical hidden scores (0.21) while DEV said +10 levels proved it.
   Defense: belief-level metrics (below) + the real games as external audit.

## Build phases
- P1 Concept inventory: formalize concepts already extracted (archetypes:
  docking/permuter/aligner/multibody/carrier; laws; hazard taxonomy;
  control mappings; goal archetypes; timers/budgets/occlusion). DONE.
- P2 Generator: arcengine games composed as (control mapping x movement
  rule x goal rule x hazard x resource/budget), parameter-randomized;
  levels escalate composition depth. DONE (tools/simgen.py, 60 games,
  tuning/holdout split, scale axis).
- P3 Failure categorizer -> BELIEF AUDIT (tools/beliefaudit.py). DONE v1.
- P4 Metric suite (below). IN PROGRESS: coining rate needs P5.
- P5 Out-of-grammar mechanics (the coining test). DONE: blind set 1
  (sx01-06, unsealed 2026-09-04 -> tuning), blind set 2 (sx07-12,
  SEALED, key in BLIND_ANSWERS_2.json). tools/blindrun.py runs the
  two-phase protocol (record beliefs, then reveal beside the record).

---

## THE ATOM (what "the core" is — 2026-09-03)

The atom is not an archetype (archetypes are molecules: hand-built,
finite, and the hidden set proved a library of them does not scale). The
atom is the LOOP:

  perceive entities at several scales
  -> generate candidate relations from a compositional grammar
  -> adopt a hypothesis on one observation
  -> test it by use; refute it on contradiction
  -> compile what survives (and log what was also consistent)

Scalability without bound comes from the hypothesis space being
GENERATIVE, not enumerated: a small grammar of relations over entities
expresses mechanics nobody hand-wrote, the way a handful of bonds
expresses every molecule.

### The loop, stated formally (v0, 2026-09-04)
State: a belief set B; each belief b = (relation r over an entity group
g, witness set S_b, status in {provisional, adopted, refuted}, logged
alternatives A_b).  Each step yields an observation o_t = (context c_t,
action a_t, outcome y_t), where the context is a small feature vector
(action-count parity, previous move axis, previous action; extensible).
  1. PREDICT y^_t from the adopted beliefs (the planner is the
     predictor: every plan step is a test — verification by use).
  2. OBSERVE y_t. A belief whose scope covers (c_t, a_t) is confirmed
     (S_b grows) or CONTRADICTED.
  3. EXPLAIN BEFORE REFUTING: on contradiction, search the conditioning
     grammar G for the simplest predicate phi such that r|phi is
     consistent with S_b + {o_t}; adopt r|phi when each side has >= k
     witnesses (k = 3) from >= 2 distinct cells; otherwise refute r and
     log it.
  4. COIN: any group with contested outcomes and no belief runs the same
     search; first-found simplest-first (positive stump before its
     complement, sorted values — deterministic).
  5. ACT: plan under adopted beliefs over the product state (cell x the
     context features any law references; teleport edges from click
     laws).  No plan = information: refute the goal (map exhausted /
     reached twice without effect), touch unvisited interactables, click
     scarce classes — never approximate the target.
Convergence contract: (i) every adopted belief is refutable by ONE
counter-example inside its scope, and plans exercise beliefs, so a
false belief with a reachable counter-example survives at most one
plan; (ii) the conditioned forms of a group are finite (|features| x
|values| x 2), each refutation removes one, so flip-flops per group are
bounded by |G|; (iii) under a null of independent outcomes a stump with
k = 3 witnesses a side is adopted by chance with probability <= 2 * 2^-6
per feature value tried (parity), and the two-cell rule removes the
dominant real-world false positive (one wall bumped twice); (iv)
exploration terminates: the frontier is finite and shrinks as terrain
colors become predicted, goal refutations are permanent for a level
except when a new law re-opens them, and laws are adopted finitely often.
Implemented: world_model.Law / _record_trial / _condition_search /
record_click / _expand; agent goal_refuted; tracker stale identity.

### Entity scales (the universality requirement: "sometimes a pixel,
sometimes an object of pixel groups")
  pixel  ->  component (same-color 4/8-connected)  ->  rigid group
  (co-moving components)  ->  class (same shape/color family)  ->
  stroke structure (thinned topology: loops, endpoints, junctions)
Control-correlation evidence picks the scale that PREDICTS best per
game; no scale is privileged a priori. (Today: component only, with
hand patches at pixel and group scale. The refactor is scheduled.)

### Relation grammar (v0 — the vocabulary the atom can coin from)
  unary  : moves-under(action), blocks, vanishes-on-contact, kills,
           mutates-state-on-contact, is-metered (budget/timer), toggles
  binary : same-as(shape|color|topology), contains, adjacent-to,
           count-equals, covers(exactly), pairs-with(port polarity)
  causal : click(X) -> delta(Y), contact(A,B) -> event
Compositions of these must be able to express every mechanic in the
DEV set; any DEV mechanic that CANNOT be expressed marks a missing
primitive (the answer-key study's job to find).

### Evidence rules (the epistemics — every one learned from a failure)
  - Adopt on one observation; persist until contradicted (Law 1).
  - Verified-by-use before spending budget on a hypothesis (V6 lesson).
  - UNREACHABLE IS INFORMATION: a plan that cannot reach its target means
    the model is incomplete — learn, do not approximate (v11 study).
  - Fail-closed: bounded spend per hypothesis per level (0.21 lesson).
  - A condemnation that blanks every goal is wrong (lethal valve).
  - Two witnesses to condemn a class (precision), with the recall cost
    now MEASURED (belief audit: holdout lethal recall fell).
  - Commit until refuted: do not re-decide among equal candidates.
  - Every adoption logs its alternatives (hypothesis ledger, agent.hlog).
  - CONTRADICTION IS A QUESTION, NOT A VERDICT (the atom v0, 2026-09-04):
    a contested belief is explained by the simplest context predicate
    that separates its successes from its failures (world_model.Law —
    a decision stump over parity / previous-move axis / previous action;
    >=3 witnesses a side, refusals from >=2 cells, refuted by one
    counter-example). The planner searches cell x context. With no law
    adopted the agent is byte-identical to before (M8-verified).
  - Commit-until-refuted needs a REFUTATION: a goal the fully explored
    map cannot reach is refuted; the alternatives get their turn.
  - Terrain testimony is the color a cell showed BEFORE you stood on it.
  - A click can testify about the body only when the body is DISTINCTIVE
    (<=2 same-colored objects); identity is revisable by non-response
    (four ignored commands while another track answers) and follows
    direct evidence (a teleport landing under a standing law).
  - Arrival without effect (twice) refutes a goal; UNREACHABLE DOES NOT
    (failure atlas 2026-09-05: refutation-by-unreachability struck the
    true goal in six games) — an unreachable candidate is deferred and
    the alternatives pursued; a frontier is where passability is
    UNPREDICTED — the exhaustive walk comes last.
  - EVIDENCE IN PROPORTION TO THE COST OF BEING WRONG (Tushar,
    2026-09-04, after V10 = 0.24): one observation may steer a cheap,
    reversible step (its next use is its test), but a UNIQUE case — a
    one-off object, a first-seen event, a singleton class — earns a
    belief only on multiple independent confirmations; until then the
    program keeps the OPEN HYPOTHESIS "this may be a mechanic I have no
    name for" alive, logs it, and spends actions to disambiguate rather
    than to exploit. Level-committing acts (condemning a class, refuting
    a goal, RESET, spending a metered budget) never rest on a single
    witness. Every one-shot commitment in the ledger backfired (lethal
    one-witness, water-as-collectible, goal flapping, arrival fake); the
    first recovery (V10) came from declining to act on approximations.
  - THE OPEN HYPOTHESIS (2026-09-05): an observation that fits no
    relation the program can name is recorded as "unknown mechanic
    here" with its context, re-tested before anything is built on it
    (two re-tests, twelve actions a level; a re-test where nothing
    moved is inconclusive, not a refutation), and confirmed by a second
    witness — spontaneous or provoked. The edge of the seen world and
    the playfield crop are hypotheses too, each tested once.
  - Negative results (refuted design hypotheses, kept): a scarce-class
    click-probe stage (preempted the class-scored click policy, r11l
    46 -> 437); appearance-first tracking under clicks (r11l 46 -> 436);
    a w == 0 floor-prediction rule (too strict under attribution noise).

### Goal induction from free information (the goal organ, 2026-09-05)

The atom's goal was "a unique small object; reach it". The failure atlas
showed that in 18 of 24 games winning is a RELATION visible on the first
frame. The organ (agent/goals.py) reads the frame into ranked GOAL
HYPOTHESES and pursues them without knowing the mechanics:

  readings  : mover -> own-colour constellation (cover / socket);
              same colour, different KIND -> align (a socket first, an
              outline next; congruent things and bars of one thickness
              are peers, never a pair; a marker 10x the mover is floor);
              fits a frame's cavity -> align; congruent with a
              constellation -> cover; a TWIN IN THE HUD names the mover
              and a unique enclosing marker is where it goes.
  effects   : learned level-wide by doing each thing once and looking —
              keys (place-free), clicks ON things (buttons, absolute),
              clicks NEAR the mover (relative offsets, transferable);
              one record per press; recent three; HUD bars ignored.
  descent   : greedy on a distance whose axes are the ones THIS mover
              has been seen to move on (only after every key had its
              turn); cell-wise cover when the marks lie within the
              mover, centre-wise when they frame a spot.
  selection : a key that moved nothing large while something small
              changed is a mode candidate; the keys that moved things
              are re-checked at once; reproduced = nothing happened,
              contradicted = the world changed hands and the record is
              rebuilt. Records are re-checked, never discarded.
  disposal  : the level ending CONFIRMS; reaching the marker with the
              level standing SATISFIES (the next candidate takes its
              turn); mover lost / budget / stalled / colour-never-moved
              REFUTE. A confirmed reading (rule + which side moves)
              goes first on the game's next level.
  hand-offs : a stalled gain key -> press what moves something else,
              retry (a precondition ran out); a refused gain key with
              the world model's body as mover -> the planner's route.
  bounds    : 20 explorations and 40 actions per hypothesis, 5
              hypotheses a level, 12 candidates, 4 per colour.

Evidence rules it added: SATISFIED NEEDS MOTION (a hypothesis whose
mover never moved under it is not satisfied by standing at the marker);
MERGED IS NOT LOST (a mover that reads as one component with its marker
is tracked by the predicted cell); A BUTTON DOES NOT MOVE (colours that
moved under our own actions are not things to press); AXES ARE WRITTEN
OFF ONLY BY THIS MOVER'S OWN RECORD (the pooled colour record carried
another thing's motion); A MODE PRESS IS RE-CHECKED, NOT TRUSTED (a
blocked arrow looks like a selection key).

Measured effect (session 23e): DEV 17 -> 22 levels; holdout L1 0/8 ->
2/8; blind set 1 L1 4/6 -> 5/6; sim tuning unchanged.

---

## MEASUREMENT: how we know the atom improves, or is flawed

Scores say whether the agent won. The atom is judged one level down, at
the level of BELIEFS, against the generator's ground truth (control
map, movement rule, goal/hazard kinds, colors are all known per game).
Tool: tools/beliefaudit.py (v1, 2026-09-03). Definitions:

M1 COINING RATE (the atom metric). Fraction of out-of-grammar mechanics
   (P5) for which the agent ends with a correct adopted relation.
   Baseline 2026-09-04, blind set 1: 0/6 (L1 cleared 3/6 by generic
   play; stagnation 370-400/400 everywhere — no vocabulary slot).
   After conditioned laws: 1/6 (sx05 alternate-axis: three laws
   adopted at steps 14/62/66, two levels in 56 actions vs 0/400).
   After causal click laws: 2/6 (sx06 click-swap: teleport law adopted
   from two witnesses; two levels in 50 actions vs one level in 308).
   After the OPEN HYPOTHESIS build (2026-09-05: anomaly ledger with
   re-tests, conditioned outcomes with the at_edge feature, edge
   probes, the playfield as a refutable hypothesis): 4/6 — sx02 wrap
   coined as an outcome law ("ACTION1 at the edge -> (15,0)"), sx01
   "pushable(14)" coined from confirmed co-motion; sx03 (mirrored twin)
   recognized as an unknown mechanic, not named. Sealed set 2 phase-1
   record: L1 cleared 4/6 (sx12 0 -> 2 levels, sx08 0 -> 1).
   Set 1 is now tuning; set 2 phase-1 record: L1 1/6, laws 0/6,
   reveal deferred to the next mechanism for a clean before/after.

M2 OBSERVATIONS TO CORRECT BELIEF. Observe-step at which the belief set
   about a concept becomes true (control map: >=3 actions learned, all
   directions true). Compare to the oracle minimum. RHAE is a speed
   metric in disguise; this is speed measured where it originates.
   Baseline 2026-09-03: median t_control = 3 (tuning), 2 (holdout).

M3 BELIEF PRECISION / RECALL vs truth, per concept: control directions,
   wall colors, collectible class, lethal class, slide classification;
   plus FALSE-PERSIST (false beliefs still held at game end — an
   epistemics flaw, never bad luck).
   Baseline (corrected instrument, 2026-09-04): control prec 1.00
   tuning / 1.00 holdout, recall ~0.97 (the earlier 0.82 "diagonal
   flaw" was a double drift subtraction in the audit — an instrument
   artifact, logged as a negative result); goal_ok 0.94/0.96; lethal
   recall is DEATHS-CONDITIONED (games with <2 deaths are undefined);
   slide_ok 0.76/0.59 under the dispersion rule; false-persist
   0.21/0.59 per game.

M4 COMPOSITION GAP (tools/simbench.py): tuning vs never-inspected
   holdout, actions-to-first-clear and L1 clear rate. A sound core
   generalizes to unseen pairings; the gap should shrink to zero.
   Baseline: tuning 26/33, holdout 18/27 games; 33 holdout levels.

M5 COMPILED-TO-INDUCED RATIO. Of the relations the agent uses to win,
   how many were hand-written (archetypes) vs coined (induced). Today
   ~all compiled; two induced relations (collectible, lethal) doubled
   the naked cold-start (2 -> 4 real levels). The research arc is this
   ratio inverting. Measured by tools/ablation.py --config naked vs
   full, and by the hlog kinds behind each win.

M6 FLIP-FLOPS. Direction/goal re-adoptions per game (a belief adopted,
   refuted, re-adopted = evidence rules wrong). Baseline 0.2 tuning /
   0.7 holdout.

M7 STAGNATION. Longest run of steps with no new belief adopted. Losing
   games sit at 380-395 of 400 steps: the hypothesis generator has run
   dry — the space is too small. This is the clearest evidence for
   building the grammar (P5). Baseline mean 165 / 190.

M8 VERSION LOCKSTEP DRIFT (tools/lockstep.py). Any behavior difference
   between agent versions on a fixed game must be attributable to a
   logged belief difference at the first divergent action. No silent
   drift, ever again (this found the v11 planner difference on mz02).

M9 EXTERNAL AUDIT. The real games (DEV + Kaggle). If M1-M8 climb while
   the games stay flat, the metrics are lying — that disagreement is
   itself the most important signal to investigate.

### Anti-Goodhart protocol
- Holdout compositions are never inspected or debugged against.
- Out-of-grammar mechanics (P5) are authored BLIND to the agent's
  current vocabulary — ideally by a separate agent/session that has not
  read the relation grammar — and their governing relation is recorded
  in the generator's metadata for M1/M3 only after the run.
- The generator is curriculum + diagnostic; the optimization target is
  the belief metrics, and the referendum is the games.

## Sequencing decision (Tushar, 2026-08-29; reaffirmed 2026-09-03)
PARALLEL, PERMANENTLY: the simulator is a living artifact updated
whenever a real game teaches something new. We do not game the system:
a low score does not condemn a concept; we learn what works. Twin
deliverables: the symbolic core (the atom) as the research thesis with
the meta-learner's compiled contribution stated plainly and measured
(M5), and a publishable paper whose spine is M1-M9 over agent versions.
