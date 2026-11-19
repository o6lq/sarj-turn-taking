from types import SimpleNamespace

from convo import brain
from convo.engine import Conversation


def call(feats, echo_flags=None):
    conv = Conversation.__new__(Conversation)
    conv.tt = SimpleNamespace(cand={"feats": feats, "start": 0})
    conv.ms = 1000
    conv.echo_flags = [False] * 50 if echo_flags is None else echo_flags
    return conv


def words(n=1, echo_ratio=0.0):
    return {"n": n, "echo_ratio": echo_ratio, "bid": 0, "content": n, "repeated_ambig": False}


def test_a_single_word_from_the_caller_can_stop_the_agent():
    assert call(words(1))._real_words()


def test_the_agents_own_voice_cannot_stop_it():
    assert not call(words(3, echo_ratio=0.8))._real_words()
    assert not call(words(2), echo_flags=[True] * 50)._real_words()


def test_one_acknowledgement_word_does_not_stop_the_demo_agent():
    for word in ("أيوه", "نعم", "شكراً", "طيب"):
        conv = call(words(1))
        conv.tt.cand["text"] = word
        assert not conv._real_words(), word
    conv = call(words(1))
    conv.tt.cand["text"] = "وقف"
    assert conv._real_words()


def test_no_words_cannot_stop_the_agent():
    assert not call(words(0))._real_words()


def after_stop(turn, quiet_ms=700, vad=0.1):
    conv = Conversation.__new__(Conversation)
    conv.tt = SimpleNamespace(turn=turn, vad=SimpleNamespace(prob=vad))
    conv.stop_ms = 10000
    conv.ms = 10000 + quiet_ms
    return conv


def test_a_short_interruption_gets_an_answer():
    assert after_stop(None)._interruption_over()


def test_a_stray_sound_after_the_stop_does_not_block_the_answer():
    assert after_stop({"voiced": 60, "last": 10000})._interruption_over()


def test_the_agent_waits_while_the_caller_is_still_talking():
    assert not after_stop(None, vad=0.9)._interruption_over()
    assert not after_stop(None, quiet_ms=300)._interruption_over()
    assert not after_stop({"voiced": 800, "last": 10600})._interruption_over()


def test_short_comma_phrases_stay_in_one_sentence():
    assert brain._SENT.match("حياك الله، حبيت أذكرك بالدفعة") is None
    assert brain._SENT.match("حياك الله، حبيت أذكرك بالدفعة. ").group(1) == "حياك الله، حبيت أذكرك بالدفعة."
