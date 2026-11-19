"""The demo's speculative reply: the brain starts during the caller's pause, but nothing plays before TAKE_TURN,
and a draft is thrown away when the caller carries on. No network: brain and voice are fakes."""
import asyncio

import numpy as np
import pytest

from convo import engine


class FakeTurnTaker:
    def __init__(self):
        self.turn = None
        self.agent_text = ""
        self.agent_active = False
        self.trace = []

    def on_scenario_start(self):
        pass

    def on_agent_start(self, text):
        self.agent_active = True

    def on_agent_end(self):
        self.agent_active = False

    def _audio_since(self, ms):
        return np.zeros(160, np.int16)


@pytest.fixture
def conv(monkeypatch):
    seen = {"brain": []}

    def sentences(msgs):
        seen["brain"].append(msgs[-1]["content"])
        yield "المبلغ ثلاث مية وخمسين ريال."

    async def tts(text, gender="f", eng="edge"):
        return np.zeros(2400, np.int16), "edge"

    monkeypatch.setattr(engine.policy, "TurnTaker", FakeTurnTaker)
    monkeypatch.setattr(engine.brain, "sentences", sentences)
    monkeypatch.setattr(engine.voice, "tts", tts)
    monkeypatch.setattr(engine, "transcribe", lambda *a, **k: pytest.fail("the turn was already read"))
    sent = []

    async def send(m):
        sent.append(m)
    c = engine.Conversation(send, "payment", "edge")
    c.state, c.phase, c.ms = "listening", "talk", 2000
    c.tt.turn = {"id": 7, "start": 500, "last": 1600, "voiced": 900, "text": "كم المبلغ؟", "text_at": 1600, "done": False}
    c.sent, c.seen = sent, seen
    return c


def audio_sent(c):
    return [m for m in c.sent if m["type"] == "say"]


def test_a_draft_plays_only_after_the_turn_is_given(conv):
    async def go():
        conv._maybe_speculate()
        assert conv.spec is not None
        await asyncio.sleep(0.2)
        assert audio_sent(conv) == []                       # written and voiced, but silent
        assert [m["role"] for m in conv.history] == ["system"]
        conv.ms = 2300
        conv.tt.turn["done"] = True
        await conv.caller_finished(conv.tt.turn["start"])
        await asyncio.sleep(0.1)
        return audio_sent(conv)
    said = asyncio.run(go())
    assert [m["text"] for m in said] == ["المبلغ ثلاث مية وخمسين ريال."]
    assert conv.seen["brain"] == ["كم المبلغ؟"]             # one brain call, from the same text
    assert [m["content"] for m in conv.history if m["role"] == "user"] == ["كم المبلغ؟"]
    reply = next(s for s in conv.stats if s["kind"] == "reply")
    assert reply["parts"]["eot_ms"] == 700 and "head_start_ms" in reply["parts"]


def test_a_draft_is_dropped_when_the_caller_carries_on(conv):
    async def go():
        conv._maybe_speculate()
        await asyncio.sleep(0.2)
        conv.tt.turn["last"] = 2100                          # they spoke again
        conv._check_spec()
        await asyncio.sleep(0.1)
    asyncio.run(go())
    assert conv.spec is None and conv.hold is None
    assert audio_sent(conv) == []
    assert [m["role"] for m in conv.history] == ["system"]


def test_no_draft_for_an_unfinished_sentence(conv):
    conv.tt.turn["text"] = "أنا كنت أبي أسأل عن يعني"
    conv._maybe_speculate()
    assert conv.spec is None


def test_no_draft_before_the_pause_is_long_enough(conv):
    conv.ms = conv.tt.turn["last"] + engine.SPECULATE_AT_MS - 20
    conv._maybe_speculate()
    assert conv.spec is None
