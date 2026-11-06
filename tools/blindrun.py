"""BLIND RUN — the coining test (M1), phase 1 of 2.

Runs the agent on the sealed out-of-grammar games (sx01..sx06; authored
by a separate session that never read the relation grammar) and writes
what the agent BELIEVED: levels cleared, the full hypothesis ledger
(agent.hlog), the final control/wall/collect/lethal beliefs, and the
stagnation curve.  It never opens environment_files/BLIND_ANSWERS.json.

Phase 2 (`--reveal`) is a separate invocation, run only AFTER phase 1
has written results/blindrun.json: it prints the sealed answers next to
the recorded beliefs so the coining verdict is made on the record, not
on a re-run tuned to the answers.  M1 = fraction of games where an
adopted relation matches the governing relation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import arc_agi  # noqa: E402

from agent.mapper_agent import MapperAgent  # noqa: E402
from agent.runner import run_episode  # noqa: E402

OUT = ROOT / "results" / "blindrun.json"
SEALED = ROOT / "environment_files" / "BLIND_ANSWERS.json"
# blind set 1 (sx01-06) was UNSEALED on 2026-09-04 and is now a tuning
# set; set 2 (sx07-12) is the live coining test until its reveal
SETS = {"1": (("sx01", "sx02", "sx03", "sx04", "sx05", "sx06"), "BLIND_ANSWERS.json"),
        "2": (("sx07", "sx08", "sx09", "sx10", "sx11", "sx12"), "BLIND_ANSWERS_2.json")}


def run(args):
    ids, _ = SETS[args.set]
    out = OUT.with_name(f"blindrun{args.set}.json")
    arc = arc_agi.Arcade(operation_mode="offline")
    rows = []
    for info in arc.get_environments():
        if not info.game_id.startswith(ids):
            continue
        ag = MapperAgent(seed=args.seed)
        stag = []
        orig = ag.observe

        def wobs(action, obs, changed, _ag=ag, _stag=stag, _orig=orig):
            r = _orig(action, obs, changed)
            _stag.append(len(_ag.hlog))
            return r
        ag.observe = wobs
        t0 = time.time()
        r = run_episode(arc, info.game_id, ag, max_actions=args.max_actions)
        m = ag.model
        # longest stretch with no new hypothesis adopted
        longest = cur = 0
        prev = None
        for k in stag:
            cur = cur + 1 if k == prev else 1
            prev = k
            longest = max(longest, cur)
        row = {
            "game": info.game_id[:4], "title": getattr(info, "title", ""),
            "levels": r["levels_completed"], "per_level": r["per_level_actions"],
            "actions": r["actions"], "secs": round(time.time() - t0, 1),
            "hlog": ag.hlog,
            "hlog_kinds": sorted({h["kind"] for h in ag.hlog}),
            "final_beliefs": {
                "motions": {a: list(v) for a, v in
                            (m.action_deltas() if m else {}).items()},
                "walls": sorted(m.wall_colors()) if m else [],
                "collect": sorted(ag._collect_colors),
                "lethal": sorted(ag._lethal_colors),
                "archetypes_active": sorted(
                    k for k, v in getattr(ag, "_gov_state", {}).items()
                    if isinstance(v, dict) and v.get("spent", 0) > 0),
            },
            "stagnation": longest, "steps": len(stag),
        }
        rows.append(row)
        print(f"  {row['game']} L={row['levels']} per={row['per_level']} "
              f"hyp={len(ag.hlog)} kinds={row['hlog_kinds']} "
              f"stag={longest}/{len(stag)} {row['secs']}s", flush=True)
    out.parent.mkdir(exist_ok=True)
    json.dump({"seed": args.seed, "max_actions": args.max_actions,
               "games": rows}, open(out, "w"), indent=1)
    print("->", out)


def reveal(args):
    ids, key = SETS[args.set]
    out = OUT.with_name(f"blindrun{args.set}.json")
    if not out.exists():
        sys.exit(f"run phase 1 first (no {out})")
    rec = json.load(open(out))
    ans = json.load(open(SEALED.with_name(key)))
    print("M1 COINING TEST — recorded beliefs vs sealed governing relations\n")
    for row in rec["games"]:
        a = ans.get(row["game"]) or ans.get(row["game"] + "-blind") or {}
        print(f"== {row['game']}  L={row['levels']} per={row['per_level']}")
        print(f"   SEALED : {json.dumps(a)[:400]}")
        print(f"   BELIEFS: {json.dumps(row['final_beliefs'])[:400]}")
        print(f"   hyp kinds: {row['hlog_kinds']}  stagnation {row['stagnation']}/{row['steps']}")
        print()
    print("Verdict per game is made by hand from this printout and written to "
          "the ledger; M1 = coined / total.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-actions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reveal", action="store_true",
                   help="phase 2: print sealed answers beside the record")
    p.add_argument("--set", default="2", choices=sorted(SETS))
    args = p.parse_args()
    if args.reveal:
        reveal(args)
    else:
        run(args)
