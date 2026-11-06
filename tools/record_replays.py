"""Record the new core playing the first level of each real game, frame
by frame with what it was doing, for the replay viewer.

  .venv/bin/python tools/record_replays.py [game ...]   -> study/replays/<game>.json
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import arc_agi
from tools.core_play2 import play_level, core_report, attempt_report

OUT = ROOT / "study" / "replays"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arc = arc_agi.Arcade(operation_mode="offline")
    envs = arc.get_environments()
    # the real games only (those with a truth file): the sim sets are not
    # to be watched, and the holdout never
    games = sys.argv[1:] or sorted(p.stem for p in (ROOT / "study" / "truth").glob("*.json"))
    for g in games:
        gid = [i.game_id for i in envs if i.game_id.startswith(g)][0]
        env = arc.make(gid); obs = env.reset()
        carry = None
        levels = tuple(range(1, int(os.environ["REPLAY_LEVELS"]) + 1)) if os.environ.get("REPLAY_LEVELS") else ((1, 2, 3) if os.environ.get("REPLAY_L3") else (1, 2))
        carries = {}
        retried = set()
        first_attempt = {}
        lv_i = 0
        while lv_i < len(levels):
            lv = levels[lv_i]
            play_level.rec = []
            t0 = time.time()
            try:
                won, steps, how, obs = play_level(env, obs, 40, (), False, carry=carry)
            except Exception as ex:
                won, steps, how = False, len(play_level.rec), f"error: {ex}"
            st = getattr(play_level, "last", {}) or {}
            # A DEATH, HONESTLY: as in the harness, one more attempt from the
            # first level carrying what the death taught; the retry is the
            # replay shown, the first attempt kept beside it
            if how == "died" and lv not in retried and st.get("model") is not None:
                retried.add(lv)
                first_attempt[lv] = {"steps": int(steps), "frames": play_level.rec}
                learned = st["model"].export(st.get("comps") or st["tr"].prev, None)
                obs = env.reset()
                carry = None
                ok_ = True
                for lv2 in levels[:lv_i]:
                    c2 = dict(carries.get(lv2) or {})
                    for k_ in ("death_touch", "touched_alive"):
                        c2[k_] = learned.get(k_, {})
                    play_level.rec = []
                    won2, steps2, how2, obs = play_level(env, obs, 40, (), False, carry=c2 if lv2 > 1 else None)
                    if not won2:
                        ok_ = False; break
                    st2 = play_level.last
                    carry = st2["model"].export(st2.get("comps") or st2["tr"].prev, st2.get("won_hyp"))
                    carry["how"] = how2.split(":")[0]
                    carry.update({k_: v_ for k_, v_ in carries.get(lv2, {}).items() if k_ in ("won_dir",)})
                if ok_:
                    carry = dict(carry or {})
                    for k_ in ("death_touch", "touched_alive", "click_gone", "click_stays", "slide_votes", "slide_against"):
                        if learned.get(k_):
                            carry[k_] = learned[k_]
                    continue
            try:
                rep = core_report(st["model"], st["tr"], st["hyps"], st["keys"], st.get("won_hyp"))
            except Exception as ex:
                rep = {"error": str(ex)}
            model = st.get("model")
            summary = model.describe() if model is not None else {}
            how_ = how if lv not in first_attempt else f"died at {first_attempt[lv]['steps']}, then {how}"
            # THE ATTEMPT LEDGER: the failure report and the divergences, for
            # the viewer to mark on the frames
            ledger_ = None
            if not won and model is not None:
                try:
                    txt_, doubts_, divs_ = attempt_report(g, lv, 1, how, st)
                    ledger_ = {"report": txt_, "doubts": [list(map(str, d_)) for d_ in doubts_],
                               "divs": [{k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in d_.items() if k_ != "law"} | {"law": str(d_.get("law"))} for d_ in divs_]}
                except Exception as ex_:
                    ledger_ = {"report": f"(ledger failed: {ex_!r})", "doubts": [], "divs": []}
            data = {"game": g if lv == 1 else f"{g} L{lv}", "won": bool(won), "steps": int(steps), "how": how_, "seconds": round(time.time() - t0, 1),
                    "frames": play_level.rec, "report": rep, "model": {k: str(v)[:300] for k, v in summary.items()},
                    "carried": bool(carry), "first_attempt": first_attempt.get(lv), "ledger": ledger_}
            (OUT / (f"{g}.json" if lv == 1 else f"{g}_L{lv}.json")).write_text(json.dumps(data, separators=(",", ":")))
            print(f"{g} L{lv}: {'WON' if won else 'lost'} in {steps} ({how_}), {len(play_level.rec)} frames", flush=True)
            play_level.rec = None
            if not won or model is None:
                break
            # the second level opens with what the first taught
            carry = model.export(st.get("comps") or st["tr"].prev, st.get("won_hyp"))
            carry["how"] = how.split(":")[0]
            try:
                h_ = st.get("won_hyp"); sw_ = getattr(play_level, "start_world", None)
                if h_ is not None and sw_ is not None:
                    tc_ = h_.target_centre(); cam_ = model.camera
                    dr_ = tc_[0] + cam_[0] - sw_[0]; dc_ = tc_[1] + cam_[1] - sw_[1]
                    carry["won_dir"] = ((dr_ > 6) - (dr_ < -6), (dc_ > 6) - (dc_ < -6))
            except Exception:
                pass
            rw = getattr(play_level, "rweights", None)
            if rw is not None:
                if st.get("won_hyp") is not None:
                    rw.won(st["won_hyp"])
                carry["relation_weights"] = rw.get()
            carries[lv] = carry
            lv_i += 1


if __name__ == "__main__":
    main()
