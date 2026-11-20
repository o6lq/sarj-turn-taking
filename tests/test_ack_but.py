"""T2 (PLAN_V6): agreement + contrast word is a floor bid. Strings written from the meaning, not from any test set."""
from turn import lexicon


def ab(t: str) -> int:
    return lexicon.analyse(t)["ack_but"]


def test_yes_but_opens_the_floor():
    for t in ("إيه بس", "أيوه بس أنا", "نعم لكن", "طيب بس عندي", "إي إي بس ودي", "تمام بس الموعد", "والله بس"):
        assert ab(t) == 1, t


def test_not_yes_but():
    for t in ("بس", "بس أبي", "إيه بس كذا", "أيوه بس هذا", "طيب بس خلاص", "إيه", "أيوه تمام",
              "المبلغ بس", "أبي بس الموعد", "شكراً"):
        assert ab(t) == 0, t
