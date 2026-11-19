"""Unit tests for the PLAN_V3 fixes, with NEW example strings (none from any test set)."""
from turn import lexicon as L


def test_asks_or_offers():
    assert L.asks_or_offers("نقدر نرسلها لك بكرة، يناسبك الوقت هذا")
    assert L.asks_or_offers("عندنا فرع العليا ولا فرع النخيل")
    assert L.asks_or_offers("وش اسم الحي؟")
    assert not L.asks_or_offers("طلبك جاهز وبيوصلك بعد يومين إن شاء الله")
