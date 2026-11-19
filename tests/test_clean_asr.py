"""clean_asr() on the kinds of text the recogniser really wrote in demo calls (docs/PLAN_V2.md §2 RC5)."""
from turn.lexicon import analyse, clean_asr


def test_speaker_labels_anywhere_are_removed():
    assert clean_asr("رغد:انا عنجد")[0] == "انا عنجد"
    text, flags = clean_asr("طيب علاء: كم المبلغ")
    assert text == "طيب كم المبلغ" and flags["labelled"]


def test_dialogue_dashes_and_repeated_labels_are_removed():
    assert clean_asr("- بدر: بدر: بدر: لحظة")[0] == "لحظة"


def test_runaway_repeats_collapse():
    assert clean_asr("هههههههه")[0] == "هه"
    assert clean_asr("طيب!!!!!!")[0] == "طيب!"
    assert clean_asr("كم؟؟؟؟")[0] == "كم؟"
    text, flags = clean_asr("سنيييييي")
    assert text == "سنيي" and flags["repeats"]


def test_subtitle_hallucinations_mean_no_speech():
    text, flags = clean_asr("شكراً جزيلاً على المشاهدة")
    assert flags["hallucination"] is False             # extra words: not the whole phrase, so it is kept
    text, flags = clean_asr("شكرا على المشاهدة")
    assert text == "" and flags["hallucination"]
    assert analyse("تصفيق")["hallucination"]


def test_fluent_formal_arabic_is_flagged():
    assert clean_asr("هل أنت متأكد أنها سوف تعود غدا إلى المنزل")[1]["fluent_msa"]
    assert not clean_asr("وش اللي صار يعني ابي افهم الحين بس")[1]["fluent_msa"]


def test_normal_speech_is_untouched():
    assert clean_asr("لا لا لحظة، كم المبلغ؟") == ("لا لا لحظة، كم المبلغ؟", {"labelled": False, "repeats": False, "hallucination": False, "fluent_msa": False})


def test_single_acknowledgement_words_are_marked():
    for word in ("أيوه", "أوكي", "أه", "ها", "شكراً", "أنا", "كان", "نعم", "إيه"):
        assert analyse(word)["single_ack"], word
    assert not analyse("لحظة")["single_ack"]
    assert not analyse("أيوه بس")["single_ack"]
