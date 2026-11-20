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
AMBIG_CONT = 1   # F5


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
    # F5: any non-backchannel word after «طيب/خلاص» (also a word repeated from the agent: «طيب الخميس») = a floor bid
    after = 0
    if AMBIG_CONT and amb:
        first = next(i for i, w in enumerate(words) if w in AMBIG)
        after = sum(1 for w in words[first + 1:] if w not in BACKCHANNEL and w not in FILLER and w not in AMBIG
                    and w not in POLITE and len(w) >= 2)
    repeated_amb = khalas >= 2 or (amb >= 1 and (content + bid) >= 1) or (AMBIG_CONT and after >= 1)
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


OFFER_CUES = {norm(w) for w in "تحب تحبين تبي تبين تبغى تبغين يناسبك يناسب تفضل تفضلين ودك تختار تختارين ولا او اختار ايهم أيهما".split()}
WEEKDAYS = {norm(w) for w in "السبت الاحد الأحد الاثنين الإثنين الثلاثاء الثلاثا الاربعاء الأربعاء الاربعا الخميس الجمعه الجمعة".split()}
DAYPARTS = {norm(w) for w in "الصبح الصباح صباحا صباحي صباحيه الصباحيه الصباحية الظهر الضهر العصر المغرب العشا العشاء الليل المسا المساء مسائي مسائيه المسائيه المسائية الضحى بكره بكرة اليوم".split()}
OPTION_HEADS = {norm(w) for w in "باقه الباقه باقة الباقة خطه الخطه فرع الفرع موعد الموعد يوم".split()}
STRONG_CHOICE = {norm(w) for w in "زين احسن أحسن خلني خليني احجز احجزه احجزها احجزلي ابي أبي ابغى أبغى ابغي ودي يناسبني مناسب افضل أفضل اخذ آخذ ناخذ اختار أختار".split()}
ACCEPT = {norm(w) for w in "ايه إيه ايوه أيوه نعم تمام اوكي ماشي طيب".split()}
_STOP_OPT = {norm(w) for w in "من في على الى إلى عن مع اللي ان إن او أو ولا و يا هذا هذي ذا".split()}


def asks_or_offers(text: str) -> bool:
    """The agent asked something or offered a choice («تحب…؟», «الصبح ولا العصر»): an answer may follow at once."""
    if "؟" in (text or "") or "?" in (text or ""):
        return True
    return any(w in OFFER_CUES for w in norm(text).split())


def _stem(w: str) -> str:
    for pre in ("وبال", "بال", "وال", "ال", "و", "ب"):
        if w.startswith(pre) and len(w) - len(pre) >= 3:
            w = w[len(pre):]
            break
    return w[:-1] if len(w) >= 4 and w.endswith(("ه", "ي")) else w


def offered_options(*agent_texts: str) -> set[str]:
    """Option slots the agent offered in an utterance that asks or offers: weekdays, parts of the day, the word after
    «باقة/فرع/موعد/يوم», and the words on both sides of «أو/ولا». Returned as stems."""
    out: set[str] = set()
    for text in agent_texts:
        if not text or not asks_or_offers(text):
            continue
        ws = norm(text).split()
        for i, w in enumerate(ws):
            if w in WEEKDAYS or w in DAYPARTS:
                out.add(_stem(w))
            if (w in OPTION_HEADS or (w.startswith("و") and w[1:] in OPTION_HEADS)) and i + 1 < len(ws) and ws[i + 1] not in _STOP_OPT and ws[i + 1] not in BID \
                    and ws[i + 1] not in OFFER_CUES and ws[i + 1] not in BACKCHANNEL:
                out.add(_stem(ws[i + 1]))
            if w in ("او", "ولا"):
                for step in (-1, 1):                  # nearest real word on each side, past «من/في/يوم…»
                    j = i + step
                    while 0 <= j < len(ws) and abs(j - i) <= 3 and (ws[j] in _STOP_OPT or ws[j] == "يوم"):
                        j += step
                    if 0 <= j < len(ws) and len(ws[j]) >= 3 and ws[j] not in BID and ws[j] not in BACKCHANNEL \
                            and ws[j] not in OFFER_CUES:
                        out.add(_stem(ws[j]))
    return out


def option_answer(caller_text: str, options: set[str]) -> dict:
    """Does the caller pick one of the offered options? {option: n words that are options, strong: a choice word
    («زين», «أحسن», «خلني»), accept: an acceptance word («إيه», «تمام»)}."""
    from difflib import SequenceMatcher
    ws = [squeeze(w) for w in norm(caller_text).split()]
    n = 0
    hits = []
    for w in ws:
        st = _stem(w)
        if len(st) >= 3 and any(st == o or (abs(len(st) - len(o)) <= 2 and SequenceMatcher(None, st, o).ratio() >= 0.8) for o in options):
            n += 1
            hits.append(w)
    return {"option": n, "hits": hits, "strong": any(w in STRONG_CHOICE for w in ws), "accept": any(w in ACCEPT for w in ws)}


# ---------------------------------------------------------------- F4: is a question/command word meant for the agent?
# question, call and command words people also say to each other or to themselves («وين حطيتي…», «شوف مين…»,
# «وش كان…؟»): they open a turn only when the words are about the call or addressed to the agent
CUE = {norm(w) for w in "وش ايش شنو شو ليش ليه متى وين كيف مين منو كم شوف شوفي اسكت اسكتي تعال تعالي".split()}
STRONG_BID = BID - CUE
# calling someone in the room (not the agent)
VOCATIVE_WORDS = {norm(w) for w in "ماما يمه يما يمي يبه يبا يبي بابا عيال ولد بنت يولد يابنت ياولد".split()}
AGENT_TITLES = {norm(w) for w in "اخي اخوي اخت اختي استاذ استاذه أستاذ أستاذة الاخ الاخت الغالي الطيب".split()}
# thinking aloud to oneself
THINK_WORDS = {norm(w) for w in """كان كانت ربي يارب يارباه خلني خليني اشوف اشوفه افكر مدري ادري اتذكر نسيت وينه وينها وينهم
شسمه اسمه اسمها ذاك ذيك حطيت حطيته حطيتها حطيتي حطيتوه وديت وديته وديتها وديتي لقيت لقيته لقيتها""".split()}
SELF_ANSWER = [norm(p) for p in ["لا خلاص", "ايه صح", "ايوه هو", "ايه هو", "ايه هذا", "اي صح", "ايوه صح", "لا لا خلاص", "تذكرت", "لقيته", "لقيتها"]]
# words about the call (money, dates, orders, accounts, appointments...) and words addressed to the agent
DOMAIN = {norm(w) for w in """ريال ريالات المبلغ مبلغ السعر سعر الاسعار الفاتوره فاتوره الرسوم رسوم الطلب طلب طلبي الطلبيه الموعد موعد موعدي
الحجز حجز التوصيل توصيل المندوب الباقه باقه باقتي الاشتراك اشتراك اشتراكي البطاقه بطاقه بطاقتي الحساب حساب حسابي الفرع فرع
الخصم خصم العرض عرض الدفعه دفعه الدفعات القسط الاقساط الشحنه شحنه الطرد السياره سيارتي الصيانه الدكتور الدكتوره العياده
النتايج النتيجه التحاليل الرقم رقم الكود الرمز العنوان عنوان التاريخ الوقت المده الشروط التفاصيل الفرق الضمان التامين
التطبيق الموقع الرابط الرساله الخدمه العقد الغاء الالغاء التجديد السداد الاسترجاع الفلوس المبالغ الشهر الاسبوع""".split()}
ADDRESS = {norm(w) for w in """انت انتي انتم عندكم عندك منكم معكم لكم لك تقول تقولين قلت قلتي تقصد تقصدين قصدك قصدكم يعني بالضبط تتصلون تتصل
اتصلتوا اتصلت ترسلون ارسلتوا تبون تبين تقدر تقدرين تقدرون عطني عطيني قولي قل وضح وضحي اشرح اشرحي عيد عيدي""".split()}
# backchannel and blessing phrases that are not content («الله يعطيك العافية» = thanks, carry on)
BC_PHRASES = [norm(p) for p in ["الحمد لله", "الحمدلله", "بيض الله وجهك", "الله يبيض وجهك", "عز الله", "عاشت ايدك", "تسلم ايدك",
                                  "عين العقل", "زين ما سويت", "الله يوفقك", "الله يسعدك", "يا هلا", "ما شاء الله", "تبارك الله", "يا سلام", "الله يعطيك العافيه", "يعطيك العافيه",
                                  "الله يجزاك خير", "جزاك الله خير", "الله يسلمك", "الله يحفظك", "الله يخليك", "صح لسانك",
                                  "على راسي", "من عيوني", "ابشر", "ابشري", "كفو", "زين والله", "والله زين", "ما قصرت",
                                  "ماقصرت", "تسلم", "تسلمين", "مشكور", "مشكوره", "الله يبارك فيك", "حياك الله"]]


_FUNCTION = {norm(w) for w in "اللي الي هذا هذي هاذا ذا ذي هذاك هذيك في من على علي الى عن مع انا هو هي احنا كذا كدا جذي الحين دحين بعد قبل عند حق".split()}


def cue_analysis(text: str, context: str = "", agent_name: str = "") -> dict:
    """F4: for a reading with question/call/command words, the cues that say whether it is for the agent.
    context = what the agent has said (previous utterance + the part of this one heard so far)."""
    t = norm(text)
    ws = [squeeze(w) for w in t.split()]
    ctx = {w for w in norm(context).split() if len(w) >= 3}
    name = norm(agent_name or "")
    strong = sum(1 for w in ws if w in STRONG_BID)
    cue = sum(1 for w in ws if w in CUE)
    voc = any(w in VOCATIVE_WORDS for w in ws)       # only clear calls to family / children, never a generic «يا …»
    others = [w for w in ws if w not in CUE and w not in BACKCHANNEL and w not in FILLER and w not in AMBIG and len(w) >= 2]
    content_words = [w for w in others if w not in _FUNCTION and w not in STRONG_BID and w not in THINK_WORDS]
    topic = any(w in DOMAIN or w.lstrip("و") in DOMAIN or (len(w) >= 4 and (w in ctx or _fuzzy_in(w, ctx)))
                or w in WEEKDAYS or w in DAYPARTS or w.isdigit() or w in ARABIC_DIGITS for w in content_words)
    address = any(w in ADDRESS for w in ws)
    think = any(w in THINK_WORDS for w in ws) or any(w in FILLER and w != "يعني" for w in ws) or any(p in t for p in SELF_ANSWER) \
        or (text or "").rstrip().endswith(("…", "..."))
    lone = cue >= 1 and not [w for w in others if w != "يعني"] and strong == 0
    return {"strong_bid": strong, "cue": cue, "vocative": int(voc), "topic": int(topic), "address": int(address),
            "think": int(think), "lone_cue": int(lone)}


def bc_phrase(text: str) -> bool:
    """The reading is a backchannel or blessing phrase («ما شاء الله، الله يعطيك العافية»): a known phrase, and at
    most two more words that carry nothing (no bid word, nothing about the call, not addressed to the agent)."""
    t = " " + norm(text) + " "
    n0 = len(t.split())
    for p in BC_PHRASES:
        t = t.replace(" " + p + " ", " ")
    if len(t.split()) == n0:
        return False
    rest = [w for w in t.split() if w not in BACKCHANNEL and w not in FILLER and w not in POLITE and w not in AMBIG]
    return len(rest) <= 2 and not any(w in STRONG_BID or w in CUE or w in DOMAIN or w in ADDRESS for w in rest)


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
