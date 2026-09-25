"""Client for the real Linguaspan ASR API (https://mlapi.linguaspanapp.com,
spec read from /openapi.json).

Auth: the API accepts either an x-api-key header or a Bearer JWT
(components.securitySchemes: ApiKeyAuth / BearerAuth). Set one of
LINGUASPAN_API_KEY / LINGUASPAN_BEARER_TOKEN -- never pass it on the command
line or paste it into chat, put it in a local .env this script loads.

POST /api/v1/transcribe takes multipart/form-data:
    audio_file (binary)  -- OR --  url (string), language (required: 'yoruba'
    | 'french' | 'english'), type ('transcription' | 'diarisation', default
    'transcription'), model_choice ('base' | 'lora', only meaningful for
    Yoruba, default 'base').
It can answer two ways (schema: anyOf[UnifiedJobResponse, TranscriptionResponse]):
    - synchronously, with the transcript straight in the response, or
    - asynchronously, with a job_id + poll_url -- this client polls
      GET /api/v1/transcribe/job/{job_id} until status is success/failed.

NOT implemented here: the presigned-S3-upload path (POST /transcribe/upload-url
then /transcribe/submit) that the docs recommend for large files -- add it if
413 "audio duration exceeds limit" errors show up against real data.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
import requests

BASE_URL = os.environ.get("LINGUASPAN_API_URL", "https://mlapi.linguaspanapp.com")
API_KEY = os.environ.get("LINGUASPAN_API_KEY", "")
BEARER_TOKEN = os.environ.get("LINGUASPAN_BEARER_TOKEN", "")
TIMEOUT_SECONDS = float(os.environ.get("LINGUASPAN_API_TIMEOUT", "120"))
POLL_INTERVAL_SECONDS = float(os.environ.get("LINGUASPAN_POLL_INTERVAL", "2"))
POLL_TIMEOUT_SECONDS = float(os.environ.get("LINGUASPAN_POLL_TIMEOUT", "300"))

# The API only accepts these three full words; map common CSV spellings onto them.
LANGUAGE_ALIASES = {
    "yo": "yoruba", "yor": "yoruba", "yoruba": "yoruba",
    "fr": "french", "fra": "french", "french": "french",
    "en": "english", "eng": "english", "english": "english",
}


def normalize_language(value: str) -> str:
    key = str(value).strip().lower()
    if key not in LANGUAGE_ALIASES:
        raise ValueError(
            f"Unrecognized language '{value}'. The API only supports "
            f"yoruba/french/english (known aliases: {sorted(LANGUAGE_ALIASES)})."
        )
    return LANGUAGE_ALIASES[key]


def load_audio_bytes(audio_ref: str) -> bytes:
    """audio_ref is a local file path or an http(s) URL; used by any client
    (Modal, generic Modal) that needs raw bytes rather than posting a URL."""
    if audio_ref.startswith("http://") or audio_ref.startswith("https://"):
        response = requests.get(audio_ref, timeout=60)
        response.raise_for_status()
        return response.content
    with open(audio_ref, "rb") as f:
        return f.read()


@dataclass
class TranscriptionResult:
    hypothesis_text: str
    confidence: float | None
    latency_ms: float
    model_name: str | None = None
    model_version: str | None = None
    language_detected: str | None = None
    audio_duration: float | None = None
    job_id: str | None = None
    raw_response: dict | None = None
    error: str | None = None


class ASRClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, bearer_token: str | None = None):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.bearer_token = bearer_token or BEARER_TOKEN
        if not self.api_key and not self.bearer_token:
            raise ValueError(
                "No credentials configured. Set LINGUASPAN_API_KEY (x-api-key) "
                "or LINGUASPAN_BEARER_TOKEN (JWT) in the environment / .env file."
            )

    def _headers(self) -> dict:
        if self.api_key:
            return {"x-api-key": self.api_key}
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def transcribe(
        self,
        audio_ref: str,
        language: str | None = None,
        model_choice: str = "base",
        processing_type: str = "transcription",
        **_ignored,
    ) -> TranscriptionResult:
        """audio_ref is a local file path (uploaded as multipart) or an
        http(s) URL (passed via the `url` field, not downloaded/re-uploaded).
        `language` is mandatory per the API schema -- pass it from a CSV
        column via evaluate.py's metadata, or a --language CLI default.
        """
        start = time.monotonic()
        try:
            if not language:
                raise ValueError(
                    "No language for this row. The API requires yoruba/french/english "
                    "per request -- add a language column to the CSV or pass --language."
                )
            data = {
                "language": normalize_language(language),
                "type": processing_type,
                "model_choice": model_choice,
            }

            if audio_ref.startswith("http://") or audio_ref.startswith("https://"):
                data["url"] = audio_ref
                response = requests.post(
                    f"{self.base_url}/api/v1/transcribe",
                    data=data, headers=self._headers(), timeout=TIMEOUT_SECONDS,
                )
            else:
                with open(audio_ref, "rb") as fh:
                    files = {"audio_file": (os.path.basename(audio_ref), fh)}
                    response = requests.post(
                        f"{self.base_url}/api/v1/transcribe",
                        data=data, files=files, headers=self._headers(), timeout=TIMEOUT_SECONDS,
                    )

            response.raise_for_status()
            payload = response.json()

            if "job_id" in payload:
                return self._poll_job(payload["job_id"], start)

            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(
                hypothesis_text=payload.get("transcription") or "",
                confidence=None,
                latency_ms=latency_ms,
                language_detected=payload.get("language"),
                audio_duration=payload.get("audio_length_seconds"),
                raw_response=payload,
            )
        except (requests.RequestException, ValueError, FileNotFoundError, OSError) as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return TranscriptionResult(hypothesis_text="", confidence=None, latency_ms=latency_ms, error=str(exc))

    def _poll_job(self, job_id: str, start_time: float) -> TranscriptionResult:
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            response = requests.get(
                f"{self.base_url}/api/v1/transcribe/job/{job_id}",
                headers=self._headers(), timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")

            if status == "success":
                latency_ms = (time.monotonic() - start_time) * 1000
                return TranscriptionResult(
                    hypothesis_text=payload.get("transcription") or "",
                    confidence=None,
                    latency_ms=latency_ms,
                    audio_duration=payload.get("audio_duration"),
                    job_id=job_id,
                    raw_response=payload,
                )
            if status == "failed":
                latency_ms = (time.monotonic() - start_time) * 1000
                return TranscriptionResult(
                    hypothesis_text="", confidence=None, latency_ms=latency_ms,
                    job_id=job_id, error=payload.get("error") or "job failed", raw_response=payload,
                )
            time.sleep(POLL_INTERVAL_SECONDS)

        latency_ms = (time.monotonic() - start_time) * 1000
        return TranscriptionResult(
            hypothesis_text="", confidence=None, latency_ms=latency_ms,
            job_id=job_id, error=f"job did not complete within {POLL_TIMEOUT_SECONDS}s",
        )


class MockASRClient(ASRClient):
    """Returns the reference text verbatim (if present) so the rest of the
    pipeline can be exercised end-to-end without hitting the real API."""

    def __init__(self):
        pass

    def transcribe(self, audio_ref: str, **params) -> TranscriptionResult:
        hypothesis = params.get("reference_text") or "the quick brown fox jumps over the lazy dog"
        return TranscriptionResult(
            hypothesis_text=hypothesis,
            confidence=1.0,
            latency_ms=0.0,
            model_name="mock",
            model_version="v0",
        )
