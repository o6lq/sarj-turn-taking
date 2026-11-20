"""The agent's live voice: Gemini writes and speaks each reply as one audio stream, so sound starts while the reply is
still being generated. One session lasts the whole call, so the model remembers what was said."""
from __future__ import annotations

import asyncio
import base64
import json
import os

import websockets

from turn import config  # noqa: F401  (loads .env)

URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
MODEL = os.environ.get("TT_LIVE_MODEL", "gemini-2.5-flash-native-audio-latest")
VOICES = {"m": "Orus", "f": "Aoede"}
RATE = 24000
CONNECT_TIMEOUT_S = 3          # was 6: a hung live voice made the reply ~11 s late
FIRST_SOUND_TIMEOUT_S = 2.5    # first sound is ~0.8 s when it works
NEXT_SOUND_TIMEOUT_S = 5


class LiveVoice:
    def __init__(self, system: str, gender: str):
        self.system = system
        self.gender = gender
        self.ws = None
        self.closed = True
        self.active = False                # the server is generating a turn
        self._replaced = False             # drop the rest of a turn that a newer one replaced
        self._cancelled = False            # drop the rest of a turn nobody is listening to
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader = None

    async def connect(self) -> None:
        self.ws = await asyncio.wait_for(websockets.connect(URL, additional_headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                                                            max_size=2 ** 24), CONNECT_TIMEOUT_S)
        speech = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES.get(self.gender, VOICES["f"])}}}
        await self.ws.send(json.dumps({"setup": {
            "model": f"models/{MODEL}", "systemInstruction": {"parts": [{"text": self.system}]}, "outputAudioTranscription": {},
            "generationConfig": {"responseModalities": ["AUDIO"], "thinkingConfig": {"thinkingBudget": 0}, "speechConfig": speech}}}))
        await asyncio.wait_for(self.ws.recv(), CONNECT_TIMEOUT_S)
        self.closed = False
        self._reader = asyncio.create_task(self._read())

    def handle(self, message: dict) -> None:
        """Route one server message to the turn that is being spoken."""
        content = message.get("serverContent") or {}
        done = bool(content.get("turnComplete"))
        if self._replaced:
            self._replaced = not done
            return
        if self._cancelled:
            if done:
                self._cancelled = self.active = False
            return
        parts = content.get("modelTurn", {}).get("parts", [])
        audio = b"".join(base64.b64decode(p["inlineData"]["data"]) for p in parts if "inlineData" in p)
        words = content.get("outputTranscription", {}).get("text", "")
        if audio or words:
            self._queue.put_nowait((audio, words))
        if done:
            self.active = False
            self._queue.put_nowait(None)

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                self.handle(json.loads(raw))
        except websockets.WebSocketException:
            pass
        finally:
            self.closed = True
            self.active = False
            self._queue.put_nowait(None)

    async def speak(self, text: str):
        """Yield (24 kHz int16 audio bytes, transcript words) until the reply ends."""
        if self.closed:
            raise ConnectionError("live voice is not connected")
        self._replaced, self._cancelled = self.active, False
        self.active = True
        self._queue = asyncio.Queue()
        await self.ws.send(json.dumps({"realtimeInput": {"text": text}}))
        timeout, spoke = FIRST_SOUND_TIMEOUT_S, False
        while True:
            item = await asyncio.wait_for(self._queue.get(), timeout)
            if item is None:
                if self.closed and not spoke:
                    raise ConnectionError("live voice closed before speaking")
                return
            spoke, timeout = True, NEXT_SOUND_TIMEOUT_S
            yield item

    def cancel(self) -> None:
        """The caller interrupted: ignore whatever is left of this reply."""
        if self.active:
            self._cancelled = True

    async def close(self) -> None:
        self.closed = True
        if self.ws is not None:
            await self.ws.close()
