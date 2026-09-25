"""Client for the GenericASR Modal class (modal_app.py) -- takes a Hugging
Face model repo ID at call time instead of one fixed model, so the UI's
"paste a model URL" field can point at anything transformers.pipeline() can
load. Same TranscriptionResult shape as ASRClient/ModalASRClient.

Requires `modal deploy modal_app.py` to have been run (adds GenericASR
alongside the existing YorubaASR in the same app).
"""
from __future__ import annotations

import time

from .api_client import TranscriptionResult, load_audio_bytes

APP_NAME = "linguaspan-asr-eval"
CLASS_NAME = "GenericASR"


class GenericModalASRClient:
    def __init__(self, model_repo: str):
        import modal
        self.model_repo = model_repo
        cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
        self._instance = cls()

    def transcribe(self, audio_ref: str, **_ignored) -> TranscriptionResult:
        start = time.monotonic()
        try:
            audio_bytes = load_audio_bytes(audio_ref)
            result = self._instance.transcribe.remote(self.model_repo, audio_bytes)
            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(
                hypothesis_text=result.get("transcription", ""),
                confidence=None,
                latency_ms=latency_ms,
                model_name=self.model_repo,
                audio_duration=result.get("audio_duration"),
                raw_response=result,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(hypothesis_text="", confidence=None, latency_ms=latency_ms, error=str(exc))
