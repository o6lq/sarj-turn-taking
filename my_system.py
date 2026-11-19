"""Our system: "who is speaking, what did they say, what does it mean". The logic lives in turn/policy.py.

    python -m ttbench.play --calls all      # official real-time check (10 min)
    python -m tools.sim                     # same scorer, in-process, seconds
"""
import os

from ttbench.server import System, run
from turn import policy

TEAM = "sarj-saud"


class MySystem(System):
    def __init__(self):
        self.tt = policy.TurnTaker()
        if os.environ.get("TT_SIM") != "1":
            from turn.asr import warmup
            warmup()                       # load Whisper before the first call, not during it

    @property
    def trace(self):
        return self.tt.trace

    def on_scenario_start(self, context):
        self.tt.on_scenario_start(context)

    def on_agent_start(self, text, duration_ms):
        self.tt.on_agent_start(text)

    def on_agent_end(self):
        self.tt.on_agent_end()

    def on_frame(self, agent, caller):
        return self.tt.on_frame(agent, caller)


if __name__ == "__main__":
    run(MySystem, TEAM, __file__)
