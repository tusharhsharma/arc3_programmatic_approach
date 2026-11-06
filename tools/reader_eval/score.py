"""Score a reader's answers against the gold readings.

    .venv/bin/python tools/reader_eval/score.py answers.json     # answers: {"<game>:<level>": "<model's raw answer>" or parsed dict}

Prints, per scene: the model's first readings (mover colour -> target
colour, kind), whether one of its first two matches the gold, the laws
it asserted, and the latency if recorded; then the totals.  Used by the
Kaggle notebook and by hand.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.reader import parse_answer


def score(dataset, answers):
    by_scene = {f"{r['game']}:{r['level']}": r for r in dataset}
    n_gold = n_hit = 0
    rows = []
    for key, ans in answers.items():
        r = by_scene.get(key)
        if r is None:
            continue
        parsed = ans if isinstance(ans, dict) else parse_answer(ans if isinstance(ans, str) else ans.get("answer", ""))
        secs = ans.get("seconds") if isinstance(ans, dict) else None
        col = {t["sid"]: t["colour"] for t in r["things"]}

        def sid(x):
            import re
            m = re.search(r"(\d+)", str(x)) if x is not None else None
            return int(m.group(1)) if m else None
        reads = []
        for rd in (parsed or {}).get("readings", [])[:4] if parsed else []:
            reads.append((rd.get("kind"), col.get(sid(rd.get("mover"))), col.get(sid(rd.get("target"))), (rd.get("why") or "")[:50]))
        gold = r.get("gold")
        hit = None
        if gold:
            n_gold += 1
            for kd, mc, tc, _ in reads[:2]:
                if mc in gold["mover"] and tc in gold["target"] and (not gold.get("kind") or kd in gold["kind"]):
                    hit = True; break
            hit = bool(hit)
            n_hit += hit
        rows.append((key, reads, (parsed or {}).get("laws"), hit, secs))
    for key, reads, laws, hit, secs in rows:
        mark = "  " if hit is None else ("OK" if hit else "--")
        print(f"{mark} {key:8} {'%5.1fs' % secs if secs else '      '} {[(k, m, t) for k, m, t, _ in reads]}" + (f" laws {laws}" if laws else ""))
    print(f"\n{n_hit}/{n_gold} gold scenes read as a person reads them; {len(rows)} scenes answered")
    return n_hit, n_gold


if __name__ == "__main__":
    ds = json.load(open(ROOT / "study" / "reader_eval" / "dataset.json"))
    ans = json.load(open(sys.argv[1]))
    score(ds, ans)
