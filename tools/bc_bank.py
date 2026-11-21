"""Backchannel probe bank S9 (PLAN T6): everyday Saudi/Gulf acknowledgement phrases while the agent talks, plus
acknowledgement + real request controls. Written by hand from SCRIPTS.md's BC/GI definitions and the meaning of the
phrases. Never from novel-test / novel-test-2 / mirror sentences. bc-dev and bc-test use DIFFERENT phrases.

    python -m tools.bc_bank                      # writes data/cp_bank_bc.json
    python -m tools.cp_gen --bank data/cp_bank_bc.json --which bc-dev  --out data/cp-bc-dev
    python -m tools.cp_gen --bank data/cp_bank_bc.json --which bc-test --out data/cp-bc-test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]

CONTEXTS = [
    {"use_case": "تذكير بدفعة (متجر نوّار)", "a0": "السلام عليكم، معك ريم من متجر نوّار. معي الأستاذ خالد؟", "u0": "إيه نعم، معك.",
     "a1": "حياك الله. حبيت أذكرك إن عندك دفعة بقيمة ثلاث مية وخمسين ريال مستحقة يوم الخميس الجاي، وتقدر تسددها من التطبيق أو من الرابط اللي بيوصلك برسالة نصية، وإذا حبيت تأجلها نرتب لك موعد ثاني بدون رسوم.",
     "anchor": "الجاي"},
    {"use_case": "تأكيد موعد (عيادات رِفاه)", "a0": "مساء الخير، معك نورة من عيادات رِفاه. أكلّم الأستاذة سارة؟", "u0": "أيوه أنا، هلا.",
     "a1": "يا هلا. حبيت أأكد موعدك يوم الأحد الساعة عشر الصبح مع الدكتورة منى في عيادة الأسنان، ونرجو الحضور قبل الموعد بربع ساعة عشان إجراءات التسجيل، وإذا عندك تأمين لا تنسين البطاقة.",
     "anchor": "الأسنان"},
    {"use_case": "موعد توصيل (مِرسال)", "a0": "هلا، معك فهد من مِرسال للتوصيل. معي الأستاذ ماجد؟", "u0": "إيه هلا والله.",
     "a1": "عندك شحنة بتوصلك بكرة بين الساعة أربع وست العصر، والمندوب بيتصل عليك قبل ما يوصل بنص ساعة، ونبي نتأكد إن العنوان حي النرجس شارع الأمير سعود فيلا اثنعش.",
     "anchor": "العصر"},
    {"use_case": "عرض باقة (سَنا للاتصالات)", "a0": "السلام عليكم، معك هند من سَنا للاتصالات. أكلّم الأستاذ ناصر؟", "u0": "وعليكم السلام، نعم.",
     "a1": "عندنا لك عرض على باقة الألياف الجديدة بسرعة خمس مية ميقا بمية وتسعة وتسعين ريال بالشهر، والتركيب مجاني، وتقدر تلغي بأي وقت بدون غرامة ولا رسوم إضافية.",
     "anchor": "بالشهر"},
    {"use_case": "استلام بطاقة (البنك)", "a0": "هلا وغلا، معك عبدالله من البنك. معي الأستاذ سلطان؟", "u0": "إيه نعم.",
     "a1": "بخصوص بطاقتك الجديدة، البطاقة جاهزة وتقدر تستلمها من أي فرع بالهوية الوطنية، وأول سنة الرسوم مجانية، وبعدها مية وخمسين ريال بالسنة، وتقدر تفعّلها من التطبيق مباشرة.",
     "anchor": "الوطنية"},
    {"use_case": "تجديد تأمين", "a0": "صباح الخير، معك لمى من شركة التأمين. أكلّم الأستاذة ريم؟", "u0": "أيوه، هلا فيك.",
     "a1": "حبيت أبلغك إن وثيقة التأمين حقتك بتنتهي آخر الشهر، والتجديد يتم من التطبيق خلال دقيقتين، والسعر نفس السنة اللي فاتت بدون أي زيادة، وفيه تغطية إضافية للأسنان.",
     "anchor": "الشهر"},
]

DIALECTS = ["sa-najdi", "gulf-kuwaiti", "sa-hijazi", "gulf-emirati", "gulf-bahraini", "sa-eastern", "gulf-qatari",
            "sa-southern", "gulf-kuwaiti", "sa-northern"]

# KEEP: acknowledgements, agreement, thanks, blessings, short confirmations of what the agent said
KEEP = {
    "bc-dev": ["إيه والله", "صح كلامك", "أكيد أكيد", "تمام تمام", "زين زين", "هيه نعم", "إي نعم إي", "أيوه أيوه",
               "واضح واضح", "حلو والله", "الله يعطيك العافية", "مشكور ما قصرت", "إن شاء الله خير", "معك معك",
               "اها اها", "أوكي أوكي", "إي صج", "مضبوط", "ماشاء الله", "يب يب"],
    "bc-test": ["إيه صح", "صدقت", "ماشي ماشي", "عدل عدل", "هيه صح", "تمام يعطيك العافية", "زين والله", "إيه إيه",
                "أكيد طبعاً", "فاهم عليك", "الله يجزاك خير", "بالضبط", "حلو حلو", "إن شاء الله", "جميل",
                "طيب حلو", "ممتاز", "إي والله", "أوكي تمام", "الحمدلله"],
}
# STOP controls: an acknowledgement, then a real request (the *marked* word is where it becomes evident)
STOP = {
    "bc-dev": ["إيه تمام [pause 250] بس *متى* يوصل بالضبط؟", "زين زين [pause 200] *لحظة* لحظة أبي أسألك",
               "إيه والله [pause 300] *كم* قلت المبلغ؟", "أوكي [pause 200] *ممكن* أغيّر الموعد؟",
               "تمام [pause 250] *لا* لا أنا ما طلبت هالشي", "أيوه أيوه [pause 200] *وش* يعني بدون رسوم؟",
               "هيه [pause 300] *اسمع* اسمع أنا مسافر", "إي نعم [pause 250] *أبي* أكلم المدير",
               "إيه [pause 200] *ليش* ما وصلتني رسالة؟", "طيب [pause 300] *عندي* سؤال لو سمحت"],
    "bc-test": ["إيه صح [pause 250] *كم* الرسوم بعدين؟", "ماشي [pause 200] *لحظة* بس أبي أتأكد",
                "تمام تمام [pause 300] *متى* أقدر أستلم؟", "زين [pause 250] *ممكن* ترسلها لي برسالة؟",
                "أوكي [pause 200] *لا* لا أنا ألغيت قبل", "إي والله [pause 300] *وين* الفرع بالضبط؟",
                "إيه إيه [pause 250] *أبغى* أعرف التفاصيل", "صدقت [pause 200] *بس* أنا مو بالرياض",
                "هيه نعم [pause 250] *ليه* السعر كذا؟", "جميل [pause 300] *اصبر* شوي خلني أشوف"],
}


def build() -> dict:
    out = {}
    for which in ("bc-dev", "bc-test"):
        calls = []
        for i, u1 in enumerate(KEEP[which] + STOP[which]):
            ctx = CONTEXTS[i % len(CONTEXTS)]
            stop = "*" in u1
            calls.append({"section": "S9", "use_case": ctx["use_case"], "dialect": DIALECTS[i % len(DIALECTS)],
                          "gender": "mf"[i % 2], "a0": ctx["a0"], "u0": ctx["u0"], "a1": [ctx["a1"]],
                          "anchor": ctx["anchor"], "anchor_mode": "end", "offset_ms": 100 + 50 * (i % 4), "u1": u1,
                          "expected": "STOP" if stop else "KEEP", "preferred": None,
                          "delivery": ("normal" if stop else ("soft", "normal")[i % 2]), "side": None, "side_who": None,
                          "sound": "none", "tv": None, "category": "GI" if stop else "BC", "probe": "S9",
                          "why": ("An acknowledgement first, then a real request: the caller takes the floor." if stop
                                  else "An everyday acknowledgement while the agent talks: the caller is following, not taking the floor.")})
        out[which] = {"S9": calls}
    out["_meta"] = {"author": "hand-written (Claude Code, T6), from SCRIPTS.md BC/GI definitions and phrase meaning",
                    "rule": "bc-dev and bc-test share no caller phrase; never taken from any test set"}
    return out


def main() -> int:
    bank = build()
    dev = {c["u1"] for c in bank["bc-dev"]["S9"]}
    test = {c["u1"] for c in bank["bc-test"]["S9"]}
    assert not dev & test, dev & test
    p = KIT / "data" / "cp_bank_bc.json"
    p.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {p}: bc-dev {len(dev)} calls, bc-test {len(test)} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
