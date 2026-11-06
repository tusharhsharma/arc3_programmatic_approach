# A reader, not a searcher — ARC Prize 2026, ARC-AGI-3 track

Tushar Sharma. Kaggle submission: notebook `tusharhsharma/arc-prize-2026-arc-agi-3-starter`,
Version 21 (bundle v38, 1 November 2026): hidden-set score **0.39**, pre-registered [0.18, 0.40].
Paper: `PAPER.md` (Paper Prize entry). Licence: MIT-0.

A programmatic agent (no learned weights, no language model, no GPU) that plays an
unfamiliar game by perceiving the frame as persistent things, reading relations between
them before the first press, fiddling in a structured way, and stating laws over kinds of
things from what each press changed. Nothing in the code names a game, a colour, a
coordinate or a threshold. Beneath it sits "the floor", a crude explorer that plays any
level the reader leaves.

Public games: 22 of 25 first levels, 44 of 183 levels; several at or under the human
baseline (ft09 all six, lp85 L1, sp80 L1, g50t L1). The three organs of the final week
(a run whose trace is a law; a lever; a rewind key and the twin that re-walks the record)
were read from the foundation's public human replays, press by press.

## Layout
- `agent/core/` — the reader: tracker (`entities.py`), scene and readings (`scene.py`),
  the progress loop (`progress.py`), the run organ (`runkey.py`), the lever and twin
  (`lever.py`), claims, journal, agents.
- `tools/core_play2.py` — the play loop and its protocols (fiddle, carry, run, twin, paint).
- `agent/` (top level) — the floor (the August explorer) and the harness adapter.
- `submission/my_agent_core_v38.py` — the exact single-file bundle submitted as Version 21;
  `tools/build_core_submission.py` rebuilds it from the modules above (`--check <game>`).
- `tools/referee_chain.sh` — the referee: 18 public first levels, 15 climbs, the twist suite.
- `tools/harness_run.py` — runs the bundle through the official ARC-AGI-3-Agents harness loop.
- `tools/press_probe.py`, `tools/run_probe.py` — diagnosis tools (they may hold known
  human presses to reach a level; the agent never sees them).
- `SUBMISSIONS.md` — the submission ledger: every row written before its score.
- `study/` — failure reports, replays and the reading notes behind each statement.

## Run
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python tools/core_play2.py sp80:1-2 --no-floor --seconds 300 -v
    .venv/bin/python tools/build_core_submission.py --check sp80

Games are fetched into `environment_files/` with the official `arc_agi` package (not
included here). The sealed holdout sets used for measurement (27 simulated games, 6 blind
games) are not part of the release: their point was never to be looked at.
