import sys, time, world as W, frozen as F, learner as L
def eff(o, a, won): return 0.0 if not won else min(1.15, (o / a) ** 2)
SETS = eval(sys.argv[1])
print("| map | game | oracle | frozen agent: won/10, mean eff | passive | v2 active (probes) | v3 cost-aware (probes) |")
print("|---|---|---|---|---|---|---|")
for mp, games in SETS:
    W.use(mp)
    for g in games:
        o = W.oracle_len(g)
        if o is None: print(f"| {mp} | {g} | unsolvable | | | | |"); continue
        fr = [F.play(g, seed=s, budget=120) for s in range(10)]
        fw = sum(r["won"] for r in fr); fe = sum(eff(o, r["actions"], r["won"]) for r in fr) / 10
        out = []
        for mk in (lambda: L.Learner(active=False), lambda: L.Learner(active=True), lambda: L.CostLearner()):
            ag = mk(); t = time.time(); r = ag.play(g)
            ne = sum(1 for x in ag.log if x[0] == "experiment")
            out.append(f"{'won' if r['won'] else 'LOST'} {r['actions']} ({eff(o, r['actions'], r['won']):.2f}), {ne}")
        print(f"| {mp} | {g} | {o} | {fw}/10, {fe:.2f} | " + " | ".join(out) + " |", flush=True)
