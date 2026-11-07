import world as W, frozen as F
import learner as L
def eff(o, a, won): return 0.0 if not won else min(1.15, (o / a) ** 2)
SETS = [("small", "dev", ["occupancy", "toggle", "timer4"]),
        ("small", "seen after v1 (not held out)", ["second_entry", "delay3", "latch", "clock", "last_move_L"]),
        ("small", "held out (after v2 freeze)", ["inverse", "brief_second", "box_only", "exits_odd"]),
        ("corridor", "held out map", ["occupancy", "toggle", "timer14", "latch", "inverse", "brief_second", "box_only"])]
print("| map | set | game | oracle | frozen: won/10, mean eff | passive: actions (eff) | active: actions (eff), experiments | active's rule | memory carried | missing variable |")
print("|---|---|---|---|---|---|---|---|---|---|")
for mp, tag, games in SETS:
    W.use(mp)
    for g in games:
        o = W.oracle_len(g)
        if o is None: print(f"| {mp} | {tag} | {g} | unsolvable | | | | | | |"); continue
        fr = [F.play(g, seed=s, budget=120) for s in range(10)]
        fw = sum(r["won"] for r in fr); fe = sum(eff(o, r["actions"], r["won"]) for r in fr) / 10
        pl = L.Learner(active=False); pa = pl.play(g)
        al = L.Learner(active=True); ac = al.play(g)
        ne = sum(1 for x in al.log if x[0] == "experiment")
        f = lambda r: f"{'won' if r['won'] else 'LOST'} {r['actions']} ({eff(o, r['actions'], r['won']):.2f})"
        print(f"| {mp} | {tag} | {g} | {o} | {fw}/10, {fe:.2f} | {f(pa)} | {f(ac)}, {ne} | {ac['rule']} | {', '.join(ac['memory'] or [])} | {'yes' if ac['missing_variable'] else 'no'} |", flush=True)
