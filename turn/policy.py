"""The turn-taking brain. Four layers, each one cheap to explain:

  1. WHO is making sound?   echo removed with the agent's clean channel, Silero VAD, and the level compared with
                            the caller's OWN voice, learned from their first answer (U0). TV, a child across the
                            room, a horn or the agent's echo are quieter than, or not, the caller's voice.
  2. WHAT did they say?     on-device Whisper on the caller's audio so far, re-read every ~250 ms.
  3. WHAT does it MEAN?     Gulf/Saudi lexicon: acknowledgement (إيه، أها، هيه) vs floor bid (لا، لحظة، بس، اسمع)
                            vs new content; words the caller repeats from the agent's own sentence = saying it along.
  4. DECIDE                 a learned score over those features (models/policy.json, trained on generated calls)
                            with transparent rules as the fallback; a sustained-loud-speech safety net.
  + END OF TURN             wait longer when the answer is unfinished (a filler, fewer digits than the agent asked for).

Every number below is a parameter the tuner (tools/tune.py) can move.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from . import acoustic, clock, decision_log, lexicon, semantic
from . import asr as _asr
from .echo import EchoSuppressor
from .vad import SileroVAD

FRAME_MS = 20
VAD_TH = 0.5             # Silero probability that counts as speech
REL_DB = 10.0            # speech must be at most this far below the caller's own level (U0)
SNR_DB = 9.0             # ... and this far above the room's noise floor
STOP_REL_DB = 8.0        # a stop needs the candidate's loudness within this of U0 (rejects side speech, TV)
MIN_VOICED_MS = 100      # voiced audio before the first ASR read
ASR_EVERY_MS = 150       # re-read the growing segment this often
GAP_END_MS = 1300        # silence that closes a candidate
CONTENT_MAX_AGE_S = 3.0  # content heard only after this much continuous sound, and quiet = background talk
MIN_LOGPROB = -0.9       # content words count only if Whisper was reasonably sure
CONTENT_MIN = 2          # new-content words that make a floor bid
LONG_MS = 2000           # safety net: sustained loud speech this long -> stop even without words
LONG_REL_DB = 3.0
SUSTAIN_MS = 0           # E3: a turn-opener first word («طيب/خلاص», a question word) + still talking this long -> stop; 0 = off
SUSTAIN_GAP_MS = 150     # ...gaps shorter than this still count as one stretch
SUSTAIN_REL_DB = 4.0     # ...at least U0 - 4 dB
SUSTAIN_SPK = 0.30       # ...and not clearly another voice
DEFAULT_REF_DB = -21.0   # caller level before U0 has been heard
AGENT_TAIL_MS = 200
EOT_BASE_MS = 600        # silence after a complete answer before TAKE_TURN
EOT_LONG_MS = 2500       # silence allowed when the answer is unfinished
NEVER_STOP = 0           # 1: only record features (used to build training data)
USE_SPEAKER = 1          # compare with the caller's own voiceprint
SPK_MIN = 0.30           # below this similarity it is someone else (TV, a child across the room)... (CAM++ scale)
SPK_LOUD_DB = -2.0       # ...unless it is clearly louder than the caller usually is
HYB_TH = 0.3            # hybrid: below this learned probability a rule-based stop is vetoed (F8: 0.2 -> 0.3)
PROMOTE_TH = 0.95        # hybrid: above this learned probability the model may stop on its own (1.01 = never)
PROMOTE_REL_DB = 4.0     # ...only when the voice is about as loud as the caller's
EOT_SMART = 0            # 1: also ask Smart Turn v3.2 (voice-based end-of-turn) before taking the turn
EOT_SMART_TH = 0.5       # below this P(complete) the caller is probably mid-thought...
EOT_MID_MS = 1500        # ...so wait this long instead of EOT_BASE_MS
USE_MODEL = 2            # 1: learned score (if models/policy.json exists), 0: rules only
MODEL_TH = 0.5
USE_SEM = 1              # level 4b: the meaning student (turn/semantic.py), if installed
SEM_LO = 0.25            # content words whose meaning says «not for the agent» (TV, echo, chatter) do not stop it
SEM_HI = 0.90            # meaning says «wants the floor» although no bid word was recognised
SEM_REL_DB = 3.0         # ...and may overrule the learned veto only for 2+ words, about as loud as the caller...
SEM_SPK = 0.35           # ...in a voice that matches the caller's voiceprint
TURN_WINDOW_MS = 20000   # turn reads look at the last 20 s: shorter reads are faster, and 20 s still holds a dictated number
RESUME_MS = 460          # take a stop back only inside this window (the scorer gives half credit within 500 ms)...
RESUME_GAP_MS = 120      # ...once the sound has ended for this long...
RESUME_MAX_VOICED_MS = 1200  # ...it was short...
RESUME_P = 0.3           # ...and the full reading says it was not a floor bid
CONFIRM_READS = 1        # E2 (rejected: +160-190 ms median latency, novel-test -4): 2 = a WEAK stop reason must be confirmed by the next read of the same candidate (1 = off)
CONFIRM_WAIT_MS = 220    # ...or, if no read comes this soon (or the voice pauses 120 ms), stop on what we have
WEAK_WHY = ("question", "model is sure")   # E2b: only the two most doubtful reasons (all four cost real stops: novel-test 84.9 -> 80.9)
OPTION_ANSWER = 1        # F2: picking an option the agent offered is a floor bid, not an echo
OPTION_EARLY_MS = 400    # ...a candidate that opened this soon after the agent started counts as an answer
Q_CONTEXT_MS = 0         # F6: this long after an agent question/offer, an answer needs less evidence...
Q_CONTENT_MIN = 1        # ...one content word
Q_SEM_HI = 0.75          # ...or a lower meaning score
HESITATION = 1           # F7: negations inside a sentence, «عندي» alone, digits alone, weighing options and
                         #     filler-led thinking are not floor bids
WEIGHING = 1             # F9: two alternatives joined by «ولا/أو», no choice word = deliberating aloud
CUE_MODE = 2             # F4 (1 = also needs topic evidence: too slow on real bids): question/call/command words stop only when about the call or addressed to the agent
CUE_SPK_VETO = 0.10      # ...and never in a voice clearly not the caller's (when the voiceprint is known)
SPK_HARD = 0.10          # any rule stop: a known voiceprint this far from the caller's = someone else...
SPK_HARD_LOUD_DB = 3.0   # ...unless it is clearly louder than the caller usually is

_MODELS = Path(__file__).resolve().parents[1] / "models"
# the learned layer is trained on the words of ONE ASR engine, so each engine has its own weights
ENGINE_POLICY = _MODELS / f"policy-{_asr.BACKEND}.json"
MODEL_PATH = ENGINE_POLICY if ENGINE_POLICY.exists() else _MODELS / "policy.json"
FEATURES = ["bid", "content", "bc", "filler", "echoed", "repeated_ambig", "ambig_alone", "question", "n",
            "voiced_s", "rel_db", "only_bc", "agent_q_prev", "since_agent_start_s", "first_word_bid", "spk_sim", "cand_age_s", "logprob", "echo_ratio",
            # levels 1-3 (how it sounds, relative to the caller's own voice) + the agent's own intonation before the overlap
            "ac_effort_rel", "ac_hf_rel", "ac_dyn_rel", "ac_decay_rel", "ac_per_rel", "ac_f0_rel", "ac_f0_range", "ac_f0_slope",
            "ac_rate_rel", "ac_voiced", "ac_far_score", "ac_arousal", "agent_slope", "sem_p"]
AC_KEYS = ["effort_rel", "hf_rel", "dyn_rel", "decay_rel", "per_rel", "f0_rel", "f0_range", "f0_slope", "rate_rel", "voiced",
           "far_score", "arousal"]


def _load_model():
    try:
        return json.loads(MODEL_PATH.read_text())
    except (OSError, ValueError):
        return None


class TurnTaker:
    def __init__(self, asr_fn=None):
        from .asr import transcribe
        from . import speaker
        self._speaker = speaker

        def job(payload):
            audio, agent_text = payload if isinstance(payload, tuple) else (payload, None)
            r = dict(asr_fn(audio) if asr_fn else transcribe(audio, priority="normal" if agent_text is None else "urgent"))
            if agent_text is not None and USE_SEM and r.get("text"):
                clean = lexicon.clean_asr(r["text"])[0]
                r["sem"] = semantic.p_stop(agent_text, clean) if clean.strip() else None
            e = speaker.embed(audio[-16000 * 3:]) if USE_SPEAKER else None
            r["spk"] = e
            r["prof"] = acoustic.segment_profile(audio[-16000 * 3:]) if len(audio) >= 1600 else None
            return r
        self.asr = clock.Async(job)
        from . import eot as _eot
        self.eot_job = clock.Async(_eot.p_complete)
        self.vad = SileroVAD()
        self.echo = EchoSuppressor()
        self.model = _load_model() if USE_MODEL else None
        self.trace: list = []          # what happened and why (for the dashboard)
        self.dlog = decision_log.get()  # TT_DECISION_LOG: the same events on disk, without audio (off by default)
        self.call_no = 0
        self.reset_call()

    # ------------------------------------------------------------------ call / agent events
    def reset_call(self):
        self.vad.reset()
        self.echo.reset()
        self.ms = 0
        self.ref_levels: list[float] = []
        self.ref_audio: list[np.ndarray] = []
        self.ref_emb = None
        self.ref_emb_n = 0
        self.ref_prof = None                 # acoustic profile of the caller's own voice (levels 1-3 reference)
        self.agent_ring = deque(maxlen=50)   # last 1 s of the agent's voice (its intonation before an overlap)
        self.ref_db = DEFAULT_REF_DB
        self.floor = deque(maxlen=250)
        self.ring = deque(maxlen=1500)      # last 30 s of caller audio, 20 ms frames
        self.agent_active = False
        self.agent_end_ms = -1e9
        self.agent_text = ""
        self.agent_start_ms = 0
        self.prev_agent_question = False
        self.stopped = False
        self.cand = None
        self.cand_id = 0
        self.turn = None                     # caller turn while the agent is silent (end-of-turn track)
        self.prev_speech = False
        self.calibrating = False             # live demo: the caller says a few words before the agent starts
        self.stop_info = None                # the last STOP, watched for a quick RESUME
        self.prev_agent_text = ""
        self.agent_utt = 0
        self.agent_name = ""
        self.q_until = -1e9

    def on_scenario_start(self, context: dict | None = None):
        self.reset_call()
        if self.dlog:
            self.call_no = self.dlog.next_call()
            self.dlog.event({"call": self.call_no, "t": 0, "kind": "call_start", "use_case": (context or {}).get("use_case", "")})

    def on_agent_start(self, text: str):
        self.prev_agent_question = self.agent_text.strip().endswith(("؟", "?")) if self.agent_text else False
        self.prev_agent_text = self.agent_text
        self.agent_active, self.agent_text, self.stopped = True, text, False
        self.agent_start_ms = self.ms
        self.turn = None
        if not self.agent_name:
            import re as _re
            m = _re.search(r"مع(?:ك|اك|كم)\s+(\S+)", text or "")
            self.agent_name = m.group(1).strip("،.") if m else ""
        self.agent_utt += 1
        if self.dlog:
            self.dlog.event({"call": self.call_no, "t": self.ms, "kind": "agent_start", "agent_utt": self.agent_utt,
                             "is_question": text.strip().endswith(("؟", "?")), "n_words": len(text.split()), "agent_text": text})

    def on_agent_end(self):
        self.agent_active = False
        self.agent_end_ms = self.ms
        if Q_CONTEXT_MS and lexicon.asks_or_offers(self.agent_text):
            self.q_until = self.ms + Q_CONTEXT_MS
        if self.dlog:
            self.dlog.event({"call": self.call_no, "t": self.ms, "kind": "agent_end", "agent_utt": self.agent_utt})

    # ------------------------------------------------------------------ helpers
    def _floor_db(self) -> float:
        return float(np.percentile(self.floor, 20)) if len(self.floor) > 25 else -70.0

    def _audio_since(self, start_ms: float) -> np.ndarray:
        n = int(max(0, (self.ms - start_ms) // FRAME_MS))
        frames = list(self.ring)[-n:] if n else []
        return np.concatenate(frames) if frames else np.zeros(0, np.int16)

    def _log(self, kind: str, **kw):
        self.trace.append({"t": self.ms, "kind": kind, **kw})
        if self.dlog:
            self.dlog.record(self.call_no, self.ms, kind, kw,
                             {"agent_active": self.agent_active, "stopped": self.stopped, "agent_utt": self.agent_utt,
                              "since_agent_start_ms": self.ms - self.agent_start_ms, "cand": (self.cand or {}).get("id")})

    # ------------------------------------------------------------------ every 20 ms
    def on_frame(self, agent: np.ndarray, caller: np.ndarray):
        self.ms += FRAME_MS
        speaking = self.agent_active or (self.ms - self.agent_end_ms) <= AGENT_TAIL_MS
        raw_db = 20 * np.log10(np.sqrt(np.mean(caller.astype(np.float64) ** 2)) / 32768 + 1e-9)
        resid_db, echo_db = self.echo.process(agent, caller, self.agent_active)
        p = self.vad.push(caller)
        self.ring.append(caller.copy())
        self.agent_ring.append(agent.copy())
        if p < 0.3:
            self.floor.append(raw_db)
        floor = self._floor_db()
        level_ok = resid_db > self.ref_db - REL_DB and resid_db > floor + SNR_DB
        # opening a candidate needs the caller's level; once open, quieter syllables of the same word still count
        is_speech = p > VAD_TH and (level_ok or (self.cand is not None and resid_db > floor + SNR_DB - 3))

        # learn the caller's own voice level from what they say while the agent is silent (U0 first)
        # (only after the agent has spoken once: before that, a TV or a room could be mistaken for the caller)
        heard_agent = self.agent_end_ms > 0 or self.calibrating
        if not speaking and heard_agent and p > 0.6 and raw_db > floor + 6:
            self.ref_levels.append(raw_db)
            if len(self.ref_levels) >= 8:
                self.ref_db = float(np.percentile(self.ref_levels[-150:], 80))
            if len(self.ref_audio) < 200:                       # up to 4 s of the caller's own voice
                self.ref_audio.append(caller.copy())
                n = len(self.ref_audio)
                if USE_SPEAKER and n >= 40 and n - self.ref_emb_n >= 25:
                    self.ref_emb = self._speaker.embed(np.concatenate(self.ref_audio))
                    self.ref_prof = acoustic.segment_profile(np.concatenate(self.ref_audio))
                    self.ref_emb_n = n

        out = None
        for tag, res, ms in self.asr.poll():
            out = self._on_asr(tag, res, ms) or out
        for tag, res, ms in self.eot_job.poll():
            if self.turn is not None and tag[1] == self.turn["id"] and tag[2] == self.turn.get("pause_id"):
                self.turn["p_complete"] = res
                self._log("smart_turn", p=None if res is None else round(res, 2))
        if speaking:
            if not self.stopped:
                out = self._while_agent_speaks(is_speech, resid_db) or out
            elif self.stop_info is not None:
                self._watch_resume(is_speech)
        else:
            self.cand = None
            out = self._end_of_turn(is_speech) or out
        self.prev_speech = is_speech
        return out

    # ------------------------------------------------------------------ core: should the agent stop?
    def _while_agent_speaks(self, is_speech: bool, db: float):
        c = self.cand
        if c is None:
            if not is_speech:
                return None
            self.cand_id += 1
            c = self.cand = {"id": self.cand_id, "start": self.ms - 120, "voiced": 0, "last": self.ms,
                             "levels": [], "sub": -1e9, "text": "", "feats": None, "agent_slope": self._agent_slope()}
            self._log("candidate", level_rel=round(db - self.ref_db, 1))
        if is_speech:
            c["run"] = c.get("run", 0) + FRAME_MS if self.ms - c["last"] <= SUSTAIN_GAP_MS + FRAME_MS else FRAME_MS
            c["voiced"] += FRAME_MS
            c["last"] = self.ms
            c["levels"].append(db)
        elif self.ms - c["last"] > GAP_END_MS:
            self._log("candidate_end", text=c["text"])
            self.cand = None
            return None
        pause_start = self.prev_speech and not is_speech
        due = self.ms - c["sub"] >= ASR_EVERY_MS
        ready = c["voiced"] >= MIN_VOICED_MS or (pause_start and c["voiced"] >= 60)
        if ready and (due or pause_start) and self.ms - c["last"] < 400:
            if self.asr.submit(("cand", c["id"], self.ms), (self._audio_since(c["start"]), self.heard_context())):
                c["sub"] = self.ms
        # safety net: long, loud, sustained speech even if the words are not understood yet
        rel = self._cand_rel(c)
        not_caller = c.get("spk") is not None and c["spk"] < SPK_MIN
        if c["voiced"] >= LONG_MS and rel >= -LONG_REL_DB and not not_caller and not (c["feats"] and c["feats"]["only_bc"]):
            return self._stop("sustained loud speech", c)
        if SUSTAIN_MS and is_speech and c.get("run", 0) >= SUSTAIN_MS and self._opener_still_talking(c, rel):
            self.stop_info = {"t": self.ms, "cid": c["id"], "strong": False, "final": False}
            return self._stop(f"opened with «{c['first_word']}» and kept talking {c['run']} ms", c)
        cf = c.get("confirm")
        if cf and (self.ms - cf["t"] >= CONFIRM_WAIT_MS or (not is_speech and self.ms - c["last"] >= 120)):
            c.pop("confirm")                     # E2: no second read in time, or the caller paused: decide on what we have
            self._log("confirm_timeout", confirm_wait_ms=self.ms - cf["t"])
            self.stop_info = {"t": self.ms, "cid": c["id"], "strong": False, "final": False}
            return self._stop(f"{cf['why']} (no second read within {self.ms - cf['t']} ms)", c)
        return None

    def _opener_still_talking(self, c: dict, rel: float) -> bool:
        """E3: the first recognised word opens a turn (ambiguous token or question word), never a backchannel, filler or
        echo; the latest read shows no call to someone else, no thinking aloud, no weighing; level and voice are the caller's."""
        w = c.get("first_word")
        if not w or not (w in lexicon.AMBIG or w in lexicon.CUE):
            return False
        f = c["feats"] or {}
        if f.get("cue_vocative") or f.get("cue_think") or f.get("cue_strong_think") or f.get("cue_weighing") or f.get("bc_phrase"):
            return False
        if rel < -SUSTAIN_REL_DB or (c.get("spk") is not None and c["spk"] < SUSTAIN_SPK):
            return False
        return True

    def _agent_slope(self) -> float:
        """Pitch slope (semitones/s) of the agent's last 600 ms: a falling agent invites a backchannel."""
        a = list(self.agent_ring)[-30:]
        if len(a) < 15:
            return 0.0
        x = np.concatenate(a)
        if np.sqrt(np.mean(x.astype(np.float64) ** 2)) < 100:
            return 0.0
        v = acoustic.segment_profile(x)["f0_slope"]
        return 0.0 if v != v else float(np.clip(v, -200, 200))

    def heard_context(self) -> str:
        """What the caller has actually heard: the previous agent sentence + the part of the current one spoken so
        far (about 2.6 words a second), not the words the agent has not reached yet."""
        words = self.agent_text.split()
        k = min(len(words), int((self.ms - self.agent_start_ms) / 1000 * 2.6) + 2)
        return " ".join((self.prev_agent_text.split() + words[:k])[-24:])

    def _option_feats(self, text: str, c: dict) -> dict:
        """F2: does the caller pick one of the options the agent offered (previous utterance, or what it has said so
        far)? Not counted when the word is only the one the agent is saying right now (a supportive repeat)."""
        if not OPTION_ANSWER:
            return {}
        words = self.agent_text.split()
        k = min(len(words), int((self.ms - self.agent_start_ms) / 1000 * 2.6) + 2)
        heard = " ".join(words[:k])
        prev_opts = lexicon.offered_options(self.prev_agent_text)
        cur_opts = lexicon.offered_options(heard)
        oa = lexicon.option_answer(text, prev_opts | cur_opts)
        if not oa["option"]:
            return {"option": 0}
        recent = {lexicon._stem(lexicon.norm(w)) for w in words[max(0, k - 4):k]}
        hit_stems = {lexicon._stem(h) for h in oa["hits"]}
        return {"option": oa["option"], "option_distinct": len(hit_stems), "choice": int(oa["strong"]), "accept": int(oa["accept"]),
                "opt_prev": int(bool(hit_stems & prev_opts)), "just_said": int(hit_stems <= recent),
                "early": int(c["start"] - self.agent_start_ms <= OPTION_EARLY_MS)}

    def _cand_rel(self, c) -> float:
        return float(np.percentile(c["levels"], 80)) - self.ref_db if c["levels"] else -99.0

    def _features(self, a: dict, c: dict) -> dict:
        return {"bid": a["bid"], "content": a["content"], "bc": a["bc"], "filler": a["filler"], "echoed": a["echoed"],
                "repeated_ambig": int(a["repeated_ambig"]), "ambig_alone": int(a["ambig"] > 0 and not a["repeated_ambig"]),
                "question": int(a["question"]), "n": a["n"], "voiced_s": c["voiced"] / 1000, "rel_db": self._cand_rel(c),
                "only_bc": int(a["n"] > 0 and a["bid"] == 0 and a["content"] == 0 and not a["repeated_ambig"]),
                "agent_q_prev": int(self.prev_agent_question),
                "since_agent_start_s": (self.ms - self.agent_start_ms) / 1000,
                "first_word_bid": int(a["first_bid"] == 0), "spk_sim": c.get("spk", 0.6),
                "cand_age_s": (self.ms - c["start"]) / 1000,
                "logprob": c.get("logprob", -1.0), "echo_ratio": a["echoed"] / max(1, a["n"]),
                **{"ac_" + k: float(np.nan_to_num(np.clip(c.get("ac", {}).get(k, 0.0), -300, 300))) for k in AC_KEYS},
                "agent_slope": c.get("agent_slope", 0.0),
                "sem_p": c.get("sem", -1.0) if c.get("sem") is not None else -1.0}

    def score(self, f: dict) -> tuple[float, str]:
        """P(the caller wants the floor). Learned weights if available, else transparent rules."""
        if f["rel_db"] < -STOP_REL_DB:
            return 0.0, "too quiet for the caller (side speech / TV)"
        if f["spk_sim"] < SPK_MIN and f["rel_db"] < SPK_LOUD_DB:
            return 0.05, f"not the caller's voice (similarity {f['spk_sim']:.2f})"
        rp, why = self._rules(f)
        if CUE_MODE and rp >= MODEL_TH and f.get("spk_known") and f["spk_sim"] < SPK_HARD and f["rel_db"] < SPK_HARD_LOUD_DB:
            return 0.1, f"{why}, but clearly not the caller's voice (similarity {f['spk_sim']:.2f})"
        if not self.model:
            return rp, why
        mp = self._model_p(f)
        if USE_MODEL == 1:                       # learned score alone
            return mp, f"model p={mp:.2f}"
        # USE_MODEL == 2 (hybrid): the rules propose, the learned model can veto an unlikely stop
        sem = f.get("sem_p", -1.0)
        sure_caller = f["n"] >= 2 and f["rel_db"] >= -SEM_REL_DB and f["spk_sim"] >= SEM_SPK
        if rp >= MODEL_TH and mp < HYB_TH and not (USE_SEM and sem >= SEM_HI and sure_caller):
            return 0.25, f"{why}, but the model doubts it (p={mp:.2f})"
        knows_caller = len(self.ref_levels) >= 8           # it may act alone only once it has heard the caller
        if rp < MODEL_TH and mp >= PROMOTE_TH and f["n"] >= 1 and f["rel_db"] > -PROMOTE_REL_DB and knows_caller \
                and not self._side_or_self(f):
            return mp, f"model is sure (p={mp:.2f}) although the words are unclear"
        return rp, f"{why} (model p={mp:.2f})"

    def _model_p(self, f: dict) -> float:
        m = self.model
        feats = m.get("features", FEATURES)
        if m.get("type") == "lgbm":
            if not hasattr(self, "_booster"):
                import lightgbm as lgb
                self._booster = lgb.Booster(model_str=m["booster"])
            return float(self._booster.predict(np.array([[f.get(k, 0.0) for k in feats]], float))[0])
        z = m["bias"] + sum(m["w"][k] * (f.get(k, 0.0) - m["mu"][k]) / (m["sd"][k] or 1.0) for k in feats)
        return float(1 / (1 + np.exp(-z)))

    def _in_question_context(self) -> bool:
        """F6: the agent asked or offered within the last 3 s (its last utterance, or the part of this one heard)."""
        if not Q_CONTEXT_MS:
            return False
        if self.ms <= self.q_until:
            return True
        words = self.agent_text.split()
        k = min(len(words), int((self.ms - self.agent_start_ms) / 1000 * 2.6) + 2)
        recent = words[max(0, k - int(Q_CONTEXT_MS / 1000 * 2.6)):k]
        return any(("؟" in w or "?" in w) for w in recent)

    def _rules(self, f: dict) -> tuple[float, str]:
        sem = f.get("sem_p", -1.0)
        q = self._in_question_context()
        cmin = Q_CONTENT_MIN if q else CONTENT_MIN
        sem_hi = Q_SEM_HI if q else SEM_HI
        has_sem = USE_SEM and sem >= 0
        if CUE_MODE and f.get("bc_phrase"):
            return 0.1, "a backchannel or blessing phrase"
        if HESITATION and f.get("cue_digits_only") and not f.get("cue_directed"):
            return 0.15, "repeating or writing down numbers"
        if HESITATION and f["bid"] >= 1 and not f.get("cue_directed") and not f.get("cue_cue") and not f.get("cue_vocative"):
            f = {**f, "bid": 0}                       # the only bid words were a mid-sentence negation or «عندي»
        if f["bid"] >= 1:
            if CUE_MODE:
                v = self._cue_verdict(f)
                if v:
                    return v
            return 1.0, "floor-bid word"
        if f["repeated_ambig"]:
            return 0.9, "«خلاص/طيب» with more after it"
        weighing = HESITATION and not f.get("choice") and (f.get("option_distinct", 0) >= 2 or f.get("cue_strong_think")
                                                          or (WEIGHING and f.get("cue_weighing")))
        if f.get("option", 0) >= 1 and not weighing and (f.get("choice") or (not f.get("just_said") and (
                f.get("accept") or f.get("early") or f.get("opt_prev")))):
            return 0.85, "answers the agent's question with an option it offered"
        if f["content"] >= cmin and has_sem and sem < SEM_LO and (not self.model or self._model_p(f) < 0.5):
            return 0.2, f"words, but the meaning is not for the agent (meaning p={sem:.2f})"
        if has_sem and sem >= sem_hi and f["n"] >= 2 and f["logprob"] >= MIN_LOGPROB and f["rel_db"] >= -3.0 \
                and f["spk_sim"] >= SEM_SPK:
            return 0.85, f"the meaning asks for the floor (meaning p={sem:.2f})"
        if f["content"] >= cmin:
            if CUE_MODE and f.get("cue_think") and not f.get("cue_address") and not f.get("cue_topic"):
                return 0.2, "thinking aloud (talking to oneself)"
            if WEIGHING and f.get("cue_weighing"):
                return 0.2, "weighing the options aloud (no choice yet)"
            if HESITATION and f.get("cue_strong_think") and not f.get("cue_address") and not f.get("cue_directed"):
                return 0.2, "thinking aloud about the call (filler-led, not addressed to the agent)"
            if f["logprob"] < MIN_LOGPROB:
                return 0.3, f"words unclear (confidence {f['logprob']:.2f})"
            if f["cand_age_s"] > CONTENT_MAX_AGE_S and f["rel_db"] < -4:
                return 0.2, "long, quiet, continuous talk without a floor bid: background"
            return 0.85, f"{f['content']} content words" + (" after an agent question" if q and f["content"] < CONTENT_MIN else "")
        if f["question"] and f["n"] >= 2:
            if has_sem and sem < SEM_LO:
                return 0.2, f"looks like a question, but the meaning is an acknowledgement (meaning p={sem:.2f})"
            return 0.8, "question"
        return 0.1, "acknowledgement / filler" if f["n"] else "no words"

    def _cue_verdict(self, f: dict):
        """F4: a bid word is a floor bid only when it is the caller talking to the agent. None = it is a bid."""
        about = f.get("cue_topic") or f.get("cue_address")
        if f.get("cue_vocative") and not about:
            return 0.1, "calls someone else in the room"
        if f.get("cue_strong_bid"):
            return None
        if f.get("spk_known") and f["spk_sim"] < CUE_SPK_VETO:
            return 0.1, f"question word in a voice that is not the caller's (similarity {f['spk_sim']:.2f})"
        if f.get("cue_think") and not f.get("cue_address"):
            return 0.15, "thinking aloud (a question word to oneself)"
        if about:
            return None
        if f.get("cue_lone_cue"):
            return 0.3, "a lone question word: waiting for more"
        if CUE_MODE == 1:
            return 0.2, "a question or command not about the call (side talk)"
        return None                     # CUE_MODE 2: no positive sign of side/self talk -> still a bid

    def _side_or_self(self, f: dict) -> bool:
        """N1: patterns the learned model must not promote to a stop on its own."""
        if not CUE_MODE:
            return False
        return bool(f.get("cue_vocative") or f.get("cue_think") or f.get("bc_phrase") or f.get("cue_lone_cue")
                    or (f.get("spk_known") and f["spk_sim"] < SPK_HARD))   # 22 % of real bids read < 0.25: only clear cases

    def _on_asr(self, tag, res, ms):
        if tag[0] == "turn":
            if self.turn is not None and tag[1] == self.turn["id"]:
                self.turn["text"] = res.get("text", "")
                self.turn["text_at"] = tag[2]       # when the audio of this read ended
                self._log("asr_turn", text=self.turn["text"], ms=round(ms))
            return None
        c = self.cand
        if c is None or tag[1] != c["id"]:
            return None
        text = res.get("text", "")
        c["logprob"] = res.get("logprob", -1.0)
        c["sem"] = res.get("sem")
        if res.get("prof"):
            c["ac"] = acoustic.relative(res["prof"], self.ref_prof)
        if res.get("spk") is not None and self.ref_emb is not None:
            c["spk"] = float(np.dot(res["spk"], self.ref_emb))
        a = lexicon.analyse(text, self.agent_text)
        if a["words"] and not c.get("first_word"):
            c["first_word"] = a["words"][0]           # E3: how this candidate opened
        f = self._features(a, c)
        f.update(self._option_feats(text, c))
        if CUE_MODE:
            ca = lexicon.cue_analysis(text, self.heard_context(), self.agent_name)
            f.update({"cue_" + k: v for k, v in ca.items()})
            f["bc_phrase"] = int(lexicon.bc_phrase(text))
            f["spk_known"] = int(c.get("spk") is not None)
        c["text"], c["feats"] = text, f
        p, why = self.score(f)
        self._log("asr", text=text, ms=round(ms), p=round(p, 2), why=why, rel_db=round(f["rel_db"], 1), f=f,
                  cand_start=c["start"])
        speaking = self.agent_active or (self.ms - self.agent_end_ms) <= AGENT_TAIL_MS
        if tag[0] == "final":
            return self._maybe_resume(p, f, why)
        if speaking and not self.stopped and p >= MODEL_TH and not NEVER_STOP:
            weak = CONFIRM_READS >= 2 and any(k in why.split(" (")[0] for k in WEAK_WHY) and f["bid"] == 0
            if weak and not c.get("confirm"):
                c["confirm"] = {"t": self.ms, "why": why}
                self._log("confirm_wait", why=why, p=round(p, 2))
                return None
            if c.get("confirm"):
                wait = self.ms - c.pop("confirm")["t"]
                why = f"{why} (confirmed after {wait} ms)"
                self._log("confirm_ok", confirm_wait_ms=wait)
            self.stop_info = {"t": self.ms, "cid": c["id"], "strong": f["bid"] >= 1 or p >= 0.95, "final": False}
            return self._stop(why, c)
        if c.get("confirm") and p < MODEL_TH:
            cf = c.pop("confirm")
            self._log("confirm_dropped", why=cf["why"], confirm_wait_ms=self.ms - cf["t"])
        return None

    # ------------------------------------------------------------------ RESUME: take back a stop that was a backchannel
    def _watch_resume(self, is_speech: bool):
        s, c = self.stop_info, self.cand
        if c is None or c["id"] != s["cid"] or s["strong"] or self.ms - s["t"] > RESUME_MS:
            self.stop_info = None if (c is None or self.ms - s["t"] > RESUME_MS) else s
            return
        if is_speech:
            c["voiced"] += FRAME_MS
            c["last"] = self.ms
            return
        if not s["final"] and self.ms - c["last"] >= RESUME_GAP_MS and c["voiced"] <= RESUME_MAX_VOICED_MS:
            if self.asr.submit(("final", c["id"], self.ms), (self._audio_since(c["start"]), self.heard_context())):
                s["final"] = True

    def _maybe_resume(self, p: float, f: dict, why: str):
        s = self.stop_info
        speaking = self.agent_active or (self.ms - self.agent_end_ms) <= AGENT_TAIL_MS
        if s is None or not self.stopped or not speaking or self.ms - s["t"] > RESUME_MS + 30:
            return None
        sem = f.get("sem_p", -1.0)
        if p < RESUME_P and f["bid"] == 0 and (sem < 0 or sem < SEM_LO) and self.ms - (self.cand or {}).get("last", 0) >= 60:
            self.stopped, self.stop_info, self.cand = False, None, None
            self._log("RESUME", why=f"it was not a floor bid after all: {why}")
            return {"decision": "RESUME", "label": "backchannel", "confidence": 0.7}
        self.stop_info = None
        return None

    def _stop(self, why, c):
        if NEVER_STOP:
            return None
        self.stopped = True
        self._log("STOP", why=why, text=c["text"])
        label = "interruption"
        return {"decision": "STOP", "label": label, "confidence": 0.8}

    # ------------------------------------------------------------------ optional: end of turn
    def _end_of_turn(self, is_speech: bool):
        t = self.turn
        if is_speech:
            if t is None:
                self.cand_id += 1
                t = self.turn = {"id": self.cand_id, "start": self.ms - 120, "last": self.ms, "text": "", "done": False,
                                 "sub": -1e9, "voiced": 0}
            t["done"] = False                  # they carried on: the turn (and its transcript) continues
            t["last"] = self.ms
            t["voiced"] += FRAME_MS
            return None
        if t is None or t["done"] or t["voiced"] < 200:
            return None
        silence = self.ms - t["last"]
        if silence >= 160 and self.ms - t["sub"] > 400 and t["sub"] < t["last"]:
            if self.asr.submit(("turn", t["id"], self.ms), self._audio_since(max(t["start"], self.ms - TURN_WINDOW_MS))):
                t["sub"] = self.ms
        if EOT_SMART and silence >= 200 and t.get("pause_id") != t["last"]:
            t["pause_id"], t["p_complete"] = t["last"], None
            self.eot_job.submit(("eot", t["id"], t["last"]), self._audio_since(max(t["start"], self.ms - 8000)))
        need = lexicon.expected_digits(self.agent_text)
        unfinished = lexicon.is_incomplete(t["text"]) or (need is not None and lexicon.count_digits(t["text"]) < need)
        wait = EOT_LONG_MS if unfinished else EOT_BASE_MS
        pc = t.get("p_complete")
        if EOT_SMART and not unfinished and pc is not None and pc < EOT_SMART_TH:
            wait = EOT_MID_MS                          # the voice says «not finished yet»
        if silence >= wait:
            t["done"] = True
            self._log("TAKE_TURN", text=t["text"], waited=silence)
            return "TAKE_TURN"
        return None
