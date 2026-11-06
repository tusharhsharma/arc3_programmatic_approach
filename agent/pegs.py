"""Peg-solitaire archetype (lf52 engine, verified empirically): pieces
sit on a lattice of identical tiles; clicking a piece SELECTS it and
clicking a cell two steps away over an adjacent piece JUMPS there,
removing the jumped piece. Win = reduce to one peg.

The mode is trigger-gated by VERIFICATION-BY-USE (the V6 lesson: an
archetype must never spend a game's click budget on an unverified
hypothesis): one cheap two-click experiment — select a peg with a legal
jump, click the landing cell — must visibly remove a peg before the
solver may engage.

Pure logic here; the mapper owns perception and execution.
"""
import sys


def jump_moves(board, pegs):
    """Legal (src, over, dst) jumps for a configuration."""
    out = []
    for p in pegs:
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            over = (p[0] + d[0], p[1] + d[1])
            dst = (p[0] + 2 * d[0], p[1] + 2 * d[1])
            if over in pegs and dst in board and dst not in pegs:
                out.append((p, over, dst))
    return out


def solve_pegs(board, pegs, node_cap=400000):
    """DFS to ONE peg (memoized on configurations); falls back to the
    deepest reduction found if solitaire-perfect is out of reach.
    Returns [(src, dst), ...] click pairs."""
    board = frozenset(board)
    seen = set()
    best = {"left": len(pegs), "path": []}
    sys.setrecursionlimit(10000)

    def dfs(state, path):
        if len(state) < best["left"]:
            best["left"] = len(state)
            best["path"] = path
        if len(state) == 1:
            return path
        if state in seen or len(seen) > node_cap:
            return None
        seen.add(state)
        for src, over, dst in jump_moves(board, state):
            ns = frozenset((state - {src, over}) | {dst})
            r = dfs(ns, path + [(src, dst)])
            if r is not None:
                return r
        return None

    full = dfs(frozenset(pegs), [])
    return full if full is not None else best["path"]
