"""A function whose attributes are per thread.

The core keeps a level's working state on its functions (`play_level.last`,
`make_successors.whys`, `plan_general.last`); in the harness several games'
cores run at once, each in its own thread, and each must see its own.
"""
import threading


class PerThread:
    def __init__(self, fn):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_tl", threading.local())
        object.__setattr__(self, "__name__", getattr(fn, "__name__", "fn"))
        object.__setattr__(self, "__doc__", getattr(fn, "__doc__", None))

    def __call__(self, *a, **k):
        return object.__getattribute__(self, "_fn")(*a, **k)

    def __getattr__(self, name):
        tl = object.__getattribute__(self, "_tl")
        try:
            return getattr(tl, name)
        except AttributeError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tl"), name, value)

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_tl"), name)
