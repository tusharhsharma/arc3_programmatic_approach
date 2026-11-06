import numpy as np

from agent.perception import Diff
from agent.world_model import FREE, UNKNOWN, WALL, CANVAS, ORIGIN, MicromouseModel, wc


def obj_w(r, c, color=7, size=1, key=None):
    return (key or (color, frozenset({(0, 0)})), color, size, (r, c))


def test_motion_learning_and_wall_marking():
    m = MicromouseModel()
    m.update_view(np.zeros((16, 16), dtype=int), (0, 0))
    # VERIFICATION BY USE: one UNCONTESTED observation is provisionally
    # known (planning with it makes execution the free confirmation);
    # a CONTESTED belief (conflicting vectors) is not known until a
    # vector reaches two votes
    d = Diff(raw_changed=2, field_changed=2, moved=[(7, 1, (0, 1))])
    m.observe_transition("ACTION4", d, [obj_w(5, 5)])
    assert m.known_motions.get("ACTION4") == (0, 1)   # provisional
    m.motion["ACTION9"].votes[(0, 1)] = 1
    m.motion["ACTION9"].votes[(1, 0)] = 1
    assert "ACTION9" not in m.known_motions           # contested: unknown
    d2 = Diff(raw_changed=2, field_changed=2, moved=[(7, 1, (0, 1))])
    m.observe_transition("ACTION4", d2, [obj_w(5, 6)])
    assert m.known_motions["ACTION4"] == (0, 1)
    assert m.avatar_pos == (5, 6)
    m.observe_transition("ACTION4", Diff(raw_changed=0), [obj_w(5, 6)])
    assert m.occ[wc((5, 7))] == WALL


def test_hud_only_change_is_treated_as_blocked():
    m = MicromouseModel()
    m.update_view(np.zeros((16, 16), dtype=int), (0, 0))
    for pos in ((2, 3), (3, 3)):
        d = Diff(raw_changed=2, field_changed=2, moved=[(7, 1, (1, 0))])
        m.observe_transition("ACTION2", d, [obj_w(*pos)])
    # raw pixels changed (HUD bar) but no world change -> wall inference fires
    hud_only = Diff(raw_changed=40, field_changed=0)
    m.observe_transition("ACTION2", hud_only, [obj_w(3, 3)])
    assert m.occ[wc((4, 3))] == WALL


def test_slide_and_drift_motion_keeps_full_vector():
    m = MicromouseModel()
    m.update_view(np.zeros((32, 32), dtype=int), (0, 0))
    for pos in ((2, 2), (2, 7)):
        d = Diff(raw_changed=10, field_changed=10, moved=[(7, 1, (0, 5))])
        m.observe_transition("ACTION4", d, [obj_w(*pos)])
    # full net vector preserved: under drift/current, action+flow IS the move
    assert m.known_motions["ACTION4"] == (0, 5)
    assert m.typical_slide("ACTION4") == 5


def test_planning_around_wall():
    m = MicromouseModel()
    m.update_view(np.zeros((5, 5), dtype=int), (0, 0))
    m.avatar_pos = (2, 0)
    m.occ[wc((2, 0))] = FREE
    for name, vec in [("ACTION1", (-1, 0)), ("ACTION2", (1, 0)),
                      ("ACTION3", (0, -1)), ("ACTION4", (0, 1))]:
        m.motion[name].votes[vec] += 3
    m.occ[wc((2, 1))] = WALL
    plan = m.plan_to((2, 2))
    assert plan is not None
    pos = (2, 0)
    for a in plan:
        dr, dc = m.known_motions[a]
        pos = (pos[0] + dr, pos[1] + dc)
        assert m.occ[wc(pos)] != WALL
    assert pos == (2, 2)


def test_color_generalization_predicts_walls():
    m = MicromouseModel()
    view = np.zeros((8, 8), dtype=int)
    view[0, :] = 9                      # a row of color-9 cells
    m.update_view(view, (0, 0))
    m.avatar_pos = (1, 1)
    m.motion["ACTION1"].votes[(-1, 0)] += 2
    m.observe_transition("ACTION1", Diff(raw_changed=0), [obj_w(1, 1)])
    assert m.occ[wc((0, 1))] == WALL
    assert 9 in m.wall_colors()
    passable = m._passable()
    assert not passable[wc((0, 5))]     # other color-9 cells predicted walls


def test_frontier_prefers_seen_unknowns():
    m = MicromouseModel()
    view = np.zeros((4, 4), dtype=int)
    m.update_view(view, (0, 0))
    for r in range(4):
        for c in range(4):
            m.occ[wc((r, c))] = FREE
    m.occ[wc((3, 3))] = UNKNOWN
    m.avatar_pos = (0, 0)
    for name, vec in [("ACTION2", (1, 0)), ("ACTION4", (0, 1))]:
        m.motion[name].votes[vec] += 1
    assert m.nearest_frontier() == (3, 3)
