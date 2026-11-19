"""What the caller's words mean for turn taking, in Saudi and Gulf Arabic.

Three groups:
  BACKCHANNEL  "I'm following, carry on"         إيه، أها، مم، زين، هيه، صح، والله ...
  FILLER       "thinking"                         اممم، ااا، يعني ...
  BID          "I want the floor"                 لا، لحظة، بس، اسمع، استنى، كم، وش، ممكن، أبي ...
Everything else is CONTENT: new information is itself a claim on the floor once there are enough words.
Words the caller repeats from the agent's own sentence (saying it along with the agent) count as backchannel.
"""
from __future__ import annotations

import re

_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w\s؀-ۿ]|[_\d٠-٩]|[،؛؟]")


def norm(t: str) -> str:
    t = _DIAC.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def squeeze(w: str) -> str:
    """ممممم -> مم, هههه -> هه, اييييه -> ايه (elongation)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", w)


BACKCHANNEL = {norm(w) for w in """
ايه ايوه ايوا ايوة ايو اي اى ايي اها اه اهاه اهه مم ممم ام هم همم هيه هييه هي يب يس
نعم اكيد اكيده طبعا تمام زين صح صحيح مضبوط بالضبط معك معاك معاي معايا وياك فاهم فاهمه فاهمك فهم فاهمين
اوكي اوك اوكيه اوكى ماشي حلو حلوه جميل ممتاز الله والله صادق صادقه عارف عارفه طيب
هه ههه هههه ههههه ههههههه
""".split()}
FILLER = {norm(w) for w in "امم اممم ااا اا ام يعني ممم اه اهه هممم".split()}
# words that open a turn, ask for one, object, or ask a question
BID = {norm(w) for w in """
لا لحظه لحظة دقيقه ثانيه ثواني استنى استني اصبر انتظر انتظري بس اسمع اسمعي اسمعني شوف شوفي
ممكن ابي ابغى ابغي ابا ودي اريد بدي عندي سؤال استفسار كم وش ايش شنو شو ليش ليه متى وين كيف مين منو اي
مو مب مش ما مافي ماني مابي مابغى
سوري عفوا لوسمحت لو سمحت سمحت يااخي اخي اخوي خويي
غلط خطا ماطلبت
لاحظه لحضه افوان عفوان سوري
عذرا اعذرني اعذريني معذره سامحني
وقف قف توقف اسكت اسكتي اقطع اقاطع اقاطعك بقاطعك بقطع
عنجد جد معقول
""".split()} - {"اي"}
WEAK_BID = {norm(w) for w in "مش ما مو مب".split()}
# «خلاص» and «طيب» are the brief's ambiguous pair: repeated or followed by more words = floor bid
AMBIG = {norm(w) for w in "خلاص طيب".split()}

# Whisper sometimes "hears" these in noise or silence: never trust them
HALLUCINATIONS = [norm(p) for p in [
    "اشتركوا في القناة", "اشترك في القناة", "شكرا للمشاهده", "ترجمه نانسي قنقر", "نانسي قنقر", "موسيقى",
    "السلام عليكم ورحمه الله", "تابعونا", "لا تنسوا الاشتراك", "سبحان الله", "قناه", "المترجم",
]]
# endings Whisper tacks on to short clips (from subtitle training data): cut them, keep the rest
TAILS = [norm(p) for p in ["شكرا لكم", "شكرا للمشاهده", "شكرا على المشاهده", "مع تحياتي", "الى اللقاء"]]
POLITE = {norm(w) for w in "شكرا مشكور مشكوره يعطيك العافيه تسلم تسلمين الله يعافيك".split()}


def _fuzzy_in(w: str, vocab: set[str]) -> bool:
    """«مكباقة» ~ «باقة», «مطبقة» ~ «مطابقة»: the agent's own words coming back distorted (echo, saying along)."""
    from difflib import SequenceMatcher
    core = w[2:] if w.startswith(("ال", "وال", "بال")) and len(w) > 4 else w
    for v in vocab:
        if len(v) >= 3 and abs(len(v) - len(core)) <= 2 and SequenceMatcher(None, core, v).ratio() >= 0.75:
            return True
    return False


def cut_loop(words: list[str]) -> list[str]:
    """Whisper sometimes gets stuck repeating: «أها، ممتازة، بس، صحيح، بس، صحيح، بس...». Everything from the
    first repeated cycle on is unreliable, so keep only what came before it."""
    for n, reps in ((1, 4), (2, 3), (3, 3)):          # «لا لا لا» is real; four of the same is a loop
        for i in range(len(words) - reps * n + 1):
            a = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == a for k in range(1, reps)):
                return words[:i]
    return words


_LABEL = re.compile(r'^\s*(?:["“«][^"”»]{1,25}["”»]\s*:|\S+(?:\s\S+)?\s+:|\S{1,12}:(?=\S|\s*$))\s*\)?')


def _strip_speaker_label(text: str) -> tuple[str, bool]:
    m = _LABEL.match(text)
    return (text[m.end():], True) if m else (text, False)


# ---------------------------------------------------------------- cleaning what speech recognition writes
_LABEL_ANYWHERE = re.compile(r"(^|\s|-)[^\s:]{1,12}\s?:(?=\s*\S)")
_DASH = re.compile(r"(^|\s)-+\s*")
_CHAR_RUN = re.compile(r"(\S)\1{3,}")
_PUNCT_RUN = re.compile(r"([!؟?.،,])\1+")
HALLUCINATION_PHRASES = [norm(p) for p in ["شكرا على المشاهدة", "شكرا للمشاهدة", "ترجمة نانسي قنقر", "اشتركوا في القناة", "تصفيق", "موسيقى"]]
GULF_MARKERS = {norm(w) for w in "وش ايش ابي ابغى ابغي يبي تبي زين الحين كذا عشان مو مب حق ترا ترى وايد شلون اللي ياخي يعطيك هلا ايه اي".split()}
MSA_MARKERS = {norm(w) for w in "إنها إنه سأعود سوف لماذا هل أنت لقد ليس لست حيث الذي التي نحن هذه".split()}
# one-word acknowledgements: never enough to stop the agent when the words are unclear
ACK_TOKENS = BACKCHANNEL | FILLER | AMBIG | POLITE | {norm(w) for w in "انا كان ها اوكي شكرا نعم ايوه اه".split()}


CLEAN_ASR = True


def clean_asr(text: str) -> tuple[str, dict]:
    """Remove what the recogniser invents (speaker labels, dialogue dashes, runaway repeats) and flag what it hallucinates."""
    flags = {"labelled": False, "repeats": False, "hallucination": False, "fluent_msa": False}
    t = text or ""
    if not CLEAN_ASR:                                 # switched off: only the old leading-label strip
        t, flags["labelled"] = _strip_speaker_label(t)
        return t, flags
    t, flags["labelled"] = _strip_speaker_label(t)      # a leading label first («أبو عابد :», «"مازن": …»)
    if _LABEL_ANYWHERE.search(t):
        flags["labelled"] = True
        t = _LABEL_ANYWHERE.sub(lambda m: m.group(1) if m.group(1) != "-" else " ", t)
    t = _DASH.sub(lambda m: m.group(1), t)
    if _CHAR_RUN.search(t):
        flags["repeats"] = True
        t = _CHAR_RUN.sub(r"\1\1", t)
    t = _PUNCT_RUN.sub(r"\1", t)
    words = t.split()
    kept: list[str] = []
    for w in words:                                   # a token repeated more than 3 times in a row: keep 2
        if len(kept) >= 3 and kept[-1] == kept[-2] == kept[-3] == w:
            flags["repeats"] = True
            continue
        kept.append(w)
    if len(kept) < len(words):
        kept = [w for i, w in enumerate(kept) if not (i >= 2 and kept[i - 1] == kept[i - 2] == w)]
    t = re.sub(r"\s+", " ", " ".join(kept)).strip()
    n = norm(t)
    if n and any(n == h or n.endswith(" " + h) for h in HALLUCINATION_PHRASES):
        flags["hallucination"] = n in HALLUCINATION_PHRASES
        if flags["hallucination"]:
            t = ""
    ws = n.split()
    flags["fluent_msa"] = len(ws) >= 6 and not any(w in GULF_MARKERS for w in ws) and any(w in MSA_MARKERS for w in ws)
    return t, flags


def analyse(text: str, agent_text: str = "") -> dict:
    """Turn the caller's words into simple, explainable features."""
    # Cohere wraps a whole reading in brackets when it is guessing at an unclear sound («(تيم هوليوود)»): unreliable
    bracketed = bool(re.fullmatch(r"\s*[\(\[].*[\)\]]\s*[:.،]?\s*", text or ""))
    raw_text = text or ""
    text, clean_flags = clean_asr(raw_text)
    if clean_flags["hallucination"] or (clean_flags["labelled"] and not norm(text)):
        bracketed = True
    # ...and invents a speaker label when guessing: «"مازن الضراب": …», «أبو عابد :», «رغد:انا…», «زي:»
    text, labelled = _strip_speaker_label(text or "")
    bracketed = bracketed or (labelled and not norm(text))
    t = norm(text)
    for tail in TAILS:
        if t.endswith(tail) and len(t) > len(tail):
            t = t[: -len(tail)].strip()
    halluc = bracketed or any(h and h in t for h in HALLUCINATIONS)
    full = [squeeze(w) for w in t.split()]
    raw = cut_loop(full)
    if len(raw) < len(full):          # the decoder looped: this whole read is unreliable
        halluc = True
    words = raw if not halluc else []
    agent_words = set(norm(agent_text).split())
    bc = fill = bid = content = echoed = amb = 0
    first_bid = None
    for i, w in enumerate(words):
        if w in BID:
            bid += 1
            first_bid = i if first_bid is None else first_bid
        elif w in AMBIG:
            amb += 1
        elif w in BACKCHANNEL or w.startswith("هه") or re.fullmatch(r"(ا|ي|ه|م)+", w or "x"):
            bc += 1
        elif w in FILLER:
            fill += 1
        elif len(w) >= 3 and (w in agent_words or _fuzzy_in(w, agent_words)):
            echoed += 1
        elif w in POLITE:
            bc += 1
        else:
            content += 1
    # «خلاص خلاص» = enough (floor bid); «طيب طيب» = okay okay (acknowledgement); either + more words = bid
    khalas = sum(1 for w in words if w == "خلاص")
    repeated_amb = khalas >= 2 or (amb >= 1 and (content + bid) >= 1)
    # «فاهم عليك» = I follow you
    if "عليك" in words and any(w in ("فاهم", "فهم", "فاهمه") for w in words) and content >= 1:
        content, bc = content - 1, bc + 1
    # reduplication of one unknown word («ماتا ماتا», «مكا مكا») is how a garbled «معك معك / هيه هيه» comes out:
    # backchannels are repeated, real content rarely is
    unknown = [w for w in words if w not in BID and w not in BACKCHANNEL and w not in FILLER and w not in AMBIG]
    if content >= 2 and bid == 0 and len(set(unknown)) == 1 and len(unknown) >= 2:
        bc, content = bc + content, 0
    # weak negations alone («مش», «ما», «مو») are usually a mis-heard «مم / ماشي»: not a bid without more words
    if bid and len(words) == 1 and words[0] in WEAK_BID:
        bid, bc = 0, bc + 1
    question = "؟" in raw_text or "?" in raw_text
    return {"words": words, "n": len(words), "bc": bc, "filler": fill, "bid": bid, "content": content,
            "echoed": echoed, "ambig": amb, "repeated_ambig": repeated_amb, "question": question,
            "first_bid": first_bid, "hallucination": halluc, "fluent_msa": clean_flags["fluent_msa"],
            "single_ack": len(words) == 1 and words[0] in ACK_TOKENS}


ARABIC_DIGITS = {norm(w): i for i, w in enumerate(
    "صفر واحد اثنين ثلاثه اربعه خمسه سته سبعه ثمانيه تسعه".split())}
ARABIC_DIGITS.update({"تلاته": 3, "اتنين": 2, "تنين": 2, "تمانيه": 8, "سابع": 7, "سبعه": 7, "ستّه": 6,
                      "اثنين": 2, "ثنين": 2, "ثلاث": 3, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "ثمان": 8, "تسع": 9})
NUMBER_WORDS = {"واحد": 1, "اثنين": 2, "ثلاث": 3, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "ثمان": 8, "تسع": 9, "عشر": 10}


OFFER_CUES = {norm(w) for w in "تحب تحبين تبي تبين تبغى تبغين يناسبك يناسب تفضل تفضلين ودك تختار تختارين ولا او اختار وش رايك ايش رايك ايهم أيهما".split()}


def asks_or_offers(text: str) -> bool:
    """The agent asked something or offered a choice («تحب…؟», «الصبح ولا العصر»): an answer may follow at once."""
    if "؟" in (text or "") or "?" in (text or ""):
        return True
    return any(w in OFFER_CUES for w in norm(text).split())


def count_digits(text: str) -> int:
    t = norm(text)
    n = sum(1 for w in t.split() if w in ARABIC_DIGITS or w.lstrip("و") in ARABIC_DIGITS)
    return n + len(re.findall(r"\d", text))


def expected_digits(agent_text: str) -> int | None:
    """«هو سبع أرقام» -> 7. The agent tells us how long the caller's answer should be."""
    t = norm(agent_text)
    for m in re.finditer(r"(\S+) (ارقام|رقم|خانات)", t):
        w = m.group(1)
        for k, v in NUMBER_WORDS.items():
            if w.startswith(k):
                return v
    return None


def is_incomplete(text: str) -> bool:
    """Ends on a word that promises more: a filler, «و», «يعني», «بس»..."""
    words = norm(text).split()
    if not words:
        return False
    last = squeeze(words[-1])
    if any(w in WAIT_WORDS for w in words[:4]) and len(words) <= 4:
        return True                    # «لحظة أشوفه» / «خلني أشوف»: they asked us to wait
    return last in FILLER or last in {"و", "يعني", "بس", "لان", "عشان", "اللي", "في", "من", "على", "مع"} or text.rstrip().endswith("…")


WAIT_WORDS = {norm(w) for w in "لحظه لحظة ثانيه دقيقه اشوف اشوفه اشوفوا خلني خليني استنى انتظر".split()}
