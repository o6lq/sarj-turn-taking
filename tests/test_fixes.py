"""Unit tests for the PLAN_V3 fixes, with NEW example strings (none from any test set)."""
from turn import lexicon as L


def test_asks_or_offers():
    assert L.asks_or_offers("نقدر نرسلها لك بكرة، يناسبك الوقت هذا")
    assert L.asks_or_offers("وش اسم الحي؟")
    assert not L.asks_or_offers("طلبك جاهز وبيوصلك بعد يومين إن شاء الله")


def test_offered_options_and_answers():
    o = L.offered_options("عندنا موعدين، يوم الأحد الضحى أو يوم الأربعاء بعد العصر، أيهم يناسبك؟")
    assert {"احد", "اربعاء"} <= o and "يوم" not in o
    assert L.offered_options("تقدر تدفع من التطبيق أو من الفرع، وش يناسبك؟") == {"تطبيق", "فرع"}
    assert L.offered_options("عندنا باقة الذهبية وباقة الفضية، تحب أيهم؟") == {"ذهبي", "فضي"}
    assert L.offered_options("نحتاج منك صورة من الهوية الوطنية وتعريف بالراتب") == set()   # nothing offered
    a = L.option_answer("إيه الأربعاء زين", o)
    assert a["option"] == 1 and a["strong"] and a["accept"]
    assert L.option_answer("إيه إيه", o)["option"] == 0
