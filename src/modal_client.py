"""Client for OUR OWN Modal deployment (modal_app.py) of the Yoruba model,
not the production asr-inference-async app. Same TranscriptionResult shape
as api_client.ASRClient, so evaluate.py/metrics.py/report.py don't care which
backend is in use.

Requires `modal deploy modal_app.py` to have been run first from this
workspace (linguanspanapp) -- that's a shared-workspace action, so it's left
as a manual step rather than something run automatically.
"""
from __future__ import annotations

import time

from .api_client import TranscriptionResult, load_audio_bytes

APP_NAME = "linguaspan-asr-eval"
CLASS_NAME = "YorubaASR"


class ModalASRClient:
    def __init__(self):
        import modal
        cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
        self._instance = cls()

    def transcribe(self, audio_ref: str, **_ignored) -> TranscriptionResult:
        """_ignored swallows language/model_choice/reference_text -- this
        client only ever calls the one Yoruba model modal_app.py deploys."""
        start = time.monotonic()
        try:
            audio_bytes = load_audio_bytes(audio_ref)
            result = self._instance.transcribe.remote(audio_bytes)
            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(
                hypothesis_text=result.get("transcription", ""),
                confidence=None,
                latency_ms=latency_ms,
                model_name="LinguaSpanApp/Yoruba-model-prod-finetunning-aug28",
                audio_duration=result.get("audio_duration"),
                raw_response=result,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(hypothesis_text="", confidence=None, latency_ms=latency_ms, error=str(exc))
