"""THE PERCEPTION-NET EXPERIMENT — can a small net learn, from the visible
games, what the hand rules say about a thing's ROLE from how it looks?

Data: every recorded first-level frame of the games that have a truth
file (study/replays/<game>.json); the label of a pixel is the role of
its colour in that game's truth (controlled, wall, floor, hud, goal,
interact, hazard, background).  Roles are per game, so the net cannot
memorise "colour 9 = controlled": it has to read shape and context.

Model: a small fully-convolutional net (one-hot colours in, role
probabilities per pixel out), ~60k parameters, CPU.

Measure: leave-games-out (5 folds of 5 games); per-pixel and per-object
accuracy on the held-out games, against the hand rules' reading of the
same objects (the tracker's tags: hud / terrain; the play's controlled).

  .venv/bin/python tools/perception_net.py            -> trains, prints the fold table
  .venv/bin/python tools/perception_net.py --save     -> also writes study/perception_net.pt
"""
import json, sys, glob, random, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLE_CLASS = {"controlled": 1, "mover": 1, "selector": 1,
              "wall": 2, "floor": 3,
              "hud": 4, "budget": 4, "decor": 4,
              "marker": 5, "socket": 5, "reference": 5, "canvas": 5,
              "button": 6, "tool": 6,
              "hazard": 7}
CLASS_NAMES = ["bg", "controlled", "wall", "floor", "hud", "goal", "interact", "hazard"]
NC = len(CLASS_NAMES)


def frame_of(hexs):
    return np.array([[int(hexs[r * 64 + c], 16) for c in range(64)] for r in range(64)], dtype=np.int64)


def load_games():
    games = {}
    for p in sorted(glob.glob(str(ROOT / "study" / "truth" / "*.json"))):
        g = Path(p).stem
        truth = json.load(open(p))
        rep = ROOT / "study" / "replays" / f"{g}.json"
        if not rep.exists():
            continue
        d = json.load(open(rep))
        frames = [frame_of(fr["frame"]) for fr in d["frames"]]
        # colour -> class (a colour with several roles: the first entity's)
        col_class = {}
        for e in truth["entities"]:
            cls = ROLE_CLASS.get(e["role"], 0)
            for c in e["colors"]:
                col_class.setdefault(int(c), cls)
        bg = Counter(frames[0].flatten().tolist()).most_common(1)[0][0]
        labels = []
        for f in frames:
            lab = np.zeros((64, 64), dtype=np.int64)
            for c, cls in col_class.items():
                lab[f == c] = cls
            lab[f == bg] = 0
            labels.append(lab)
        games[g] = (frames, labels, col_class, bg)
    return games


def one_hot(f):
    x = np.zeros((16, 64, 64), dtype=np.float32)
    for c in range(16):
        x[c] = (f == c)
    return x


def build_net():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=8, dilation=8), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=16, dilation=16), nn.ReLU(),
        nn.Conv2d(32, NC, 1),
    )


def train(games, train_names, epochs=12, seed=0):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); random.seed(seed)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xs, ys = [], []
    for g in train_names:
        frames, labels, _, _ = games[g]
        # every 2nd frame: consecutive frames are near-duplicates
        for f, lab in list(zip(frames, labels))[::2]:
            xs.append(one_hot(f)); ys.append(lab)
    X = torch.tensor(np.stack(xs)); Y = torch.tensor(np.stack(ys))
    # class weights: background dominates
    counts = np.bincount(Y.numpy().flatten(), minlength=NC).astype(np.float64) + 1
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            xb, yb = X[idx], Y[idx]
            # augmentation: the eight symmetries of the square
            k = random.randint(0, 3); flip = random.random() < 0.5
            xb = torch.rot90(xb, k, (2, 3)); yb = torch.rot90(yb, k, (1, 2))
            if flip:
                xb = torch.flip(xb, (3,)); yb = torch.flip(yb, (2,))
            out = net(xb)
            loss = F.cross_entropy(out, yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
    return net


def evaluate(net, games, test_names):
    """Per-pixel and per-object accuracy on held-out games, plus the hand
    rules' object accuracy on the same objects (tracker tags -> roles)."""
    import torch
    from agent.core.scene import SceneTracker
    res = {}
    for g in test_names:
        frames, labels, col_class, bg = games[g]
        f, lab = frames[0], labels[0]
        with torch.no_grad():
            pred = net(torch.tensor(one_hot(f))[None])[0].argmax(0).numpy()
        mask = lab > 0
        pix = float((pred[mask] == lab[mask]).mean()) if mask.any() else float("nan")
        # objects on the first frame: the net's majority vote vs the truth
        tr = SceneTracker(); tr.update(f, bg)
        obj_ok = obj_n = hand_ok = 0
        per_class = Counter(); per_class_ok = Counter()
        for s in tr.prev:
            if s.sid >= 1000 or s.size < 2:
                continue
            cells = list(s.cells)
            truth_cls = Counter(int(lab[r, c]) for r, c in cells).most_common(1)[0][0]
            if truth_cls == 0:
                continue
            net_cls = Counter(int(pred[r, c]) for r, c in cells).most_common(1)[0][0]
            # the hand rules' reading of the same object
            if "hud" in s.tags or "bar" in s.tags:
                hand_cls = 4
            elif "terrain" in s.tags:
                hand_cls = 2 if s.body not in (bg,) and s.size > 300 else 3
            else:
                hand_cls = -1                      # (a thing: the rules say nothing of its role before a press)
            obj_n += 1; per_class[truth_cls] += 1
            if net_cls == truth_cls:
                obj_ok += 1; per_class_ok[truth_cls] += 1
            if hand_cls == truth_cls:
                hand_ok += 1
        res[g] = {"pixel": pix, "objects": obj_n, "net_ok": obj_ok, "hand_ok": hand_ok,
                  "per_class": {CLASS_NAMES[k]: (per_class_ok[k], per_class[k]) for k in per_class}}
    return res


def main():
    games = load_games()
    names = sorted(games)
    print(f"{len(names)} games, {sum(len(v[0]) for v in games.values())} frames")
    random.seed(1); order = names[:]; random.shuffle(order)
    folds = [order[i::5] for i in range(5)]
    all_res = {}
    t0 = time.time()
    for i, test in enumerate(folds):
        train_names = [g for g in names if g not in test]
        net = train(games, train_names)
        all_res.update(evaluate(net, games, test))
        print(f"fold {i}: trained on {len(train_names)}, tested on {test} ({time.time() - t0:.0f}s)", flush=True)
    print()
    print(f"{'game':6} {'pixel':>6} {'objs':>5} {'net':>5} {'hand':>5}  per class (net ok / n)")
    tot = Counter()
    per_cls = Counter(); per_cls_ok = Counter()
    for g in names:
        r = all_res[g]
        tot["objs"] += r["objects"]; tot["net"] += r["net_ok"]; tot["hand"] += r["hand_ok"]
        for k, (ok, n) in r["per_class"].items():
            per_cls[k] += n; per_cls_ok[k] += ok
        print(f"{g:6} {r['pixel']:6.2f} {r['objects']:5d} {r['net_ok']:5d} {r['hand_ok']:5d}  " + " ".join(f"{k}:{ok}/{n}" for k, (ok, n) in r["per_class"].items()))
    print(f"\nobjects {tot['objs']}: net right {tot['net']} ({tot['net'] / max(1, tot['objs']):.2f}), hand rules right {tot['hand']} ({tot['hand'] / max(1, tot['objs']):.2f})")
    print("per class:", {k: f"{per_cls_ok[k]}/{per_cls[k]}" for k in per_cls})
    if "--save" in sys.argv:
        import torch
        net = train(games, names)
        torch.save(net.state_dict(), ROOT / "study" / "perception_net.pt")
        print("saved study/perception_net.pt")


if __name__ == "__main__":
    main()
