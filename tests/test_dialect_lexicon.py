"""T1 (PLAN_V6): dialect words. Strings written from dialect meaning, not from any test set."""
from turn import lexicon as L


def test_inzeen_is_ambiguous_like_tayyib():
    a = L.analyse("انزين")
    assert a["n"] == 1 and a["bid"] == 0          # alone it is ambiguous (like «طيب»), not a bid
    assert "انزين" in L.AMBIG and "زين" not in L.AMBIG   # «زين» stays a backchannel


def test_gulf_hijazi_question_words_are_cues_not_strong_bids():
    for w in ["شلون", "شقد", "شكثر", "فين", "امتى"]:
        assert L.norm(w) in L.CUE and L.norm(w) not in L.STRONG_BID
        assert L.cue_analysis(f"{w} الطلب؟")["cue"] >= 1
    assert L.cue_analysis("الطلب وصل")["cue"] == 0


def test_whats_it_called_is_hesitation():
    for t in ["إيش اسمه", "ايش اسمو", "شنو اسمه", "شو اسمها", "وش اسمه", "ويش اسمه", "وشسمه"]:
        ca = L.cue_analysis(t + " الشي")
        assert ca["think"] == 1 and ca["cue"] == 0, t
    # «اسمك» = asking for a name: a real question, not a filler
    assert L.cue_analysis("شو اسمك")["cue"] >= 1


def test_hijazi_wait_is_a_strong_bid():
    assert L.norm("استنا") in L.STRONG_BID
    assert L.norm("استنى") in L.STRONG_BID
