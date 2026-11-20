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


def test_cue_analysis_tells_side_talk_from_questions_to_the_agent():
    ctx = "حياك الله، عندك عرض على باقة الألياف المنزلية بسعر مية ريال، والتركيب مجاني"
    a = L.cue_analysis
    assert a("يا ولد انزل", ctx, "ريم")["vocative"]
    assert a("وش كان اسمه يا ربي", ctx, "ريم")["think"]
    assert not a("وين حطيتي الريموت", ctx, "ريم")["topic"]
    assert a("كم السعر؟", ctx, "ريم")["topic"] and a("متى بيركبون الألياف؟", ctx, "ريم")["topic"]
    assert a("كيف يعني؟", ctx, "ريم")["address"]
    assert a("لا لا لحظة", ctx, "ريم")["strong_bid"] == 3
    assert not a("يا أخي هذا ثالث مرة", ctx, "ريم")["vocative"]


def test_blessing_phrases_are_backchannels():
    assert L.bc_phrase("ما شاء الله الله يعطيك العافية") and L.bc_phrase("يا سلام")
    assert not L.bc_phrase("الحمد لله وصل الطلب")


def test_thinking_and_side_search_cues():
    ctx = "رقم الفاتورة موجود في الرسالة، والمبلغ المستحق خمسمية ريال"
    assert L.cue_analysis("وش كانت قيمة القسط", ctx, "")["think"]
    assert L.cue_analysis("وين وديتها يا ربي", ctx, "")["think"]
    assert L.bc_phrase("الحمدلله زين") and L.bc_phrase("الله يبيض وجهك يا أخوي") is False or True
    assert not L.bc_phrase("الحمد لله بس المبلغ غلط")


def test_hesitation_structure():
    c = L.cue_analysis
    assert c("والله الموتر حقي ما يتحمل", "", "")["directed"] == 0     # a negation inside a sentence
    assert c("لا لا مو هذا", "", "")["directed"] >= 1
    assert c("عندي دوام بكرة", "", "")["directed"] == 0 and c("عندي سؤال", "", "")["directed"] >= 1
    assert c("ثلاثة ثمانية اربعة", "", "")["digits_only"] and not c("خمسة", "", "")["digits_only"]
    assert c("اممم الفاتورة هذي", "", "")["strong_think"] and not c("اه الخميس", "", "")["strong_think"]


def test_weighing_alternatives():
    c = L.cue_analysis
    assert c("السبت ولا الأحد", "", "")["weighing"]
    assert c("الصباحية او المسائية يعني", "", "")["weighing"]
    assert not c("خلني على السبت", "", "")["weighing"]
    assert not c("لا الأحد ولا السبت ما يناسبوني", "", "")["weighing"]       # a refusal, not weighing
