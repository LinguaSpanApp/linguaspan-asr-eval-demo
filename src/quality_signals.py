"""Reference-free automated quality signals (doc section 9 / 7.3 trace events).

These run on every row regardless of whether a reference transcript exists --
this is the fallback the doc calls out explicitly: "Production audio generally
arrives without a verified reference transcript, so conventional WER cannot be
calculated for every request." Signals only fire when the input they need is
present (e.g. duration_anomaly needs an audio_duration column); missing inputs
just mean that signal is skipped for that row, not an error.
"""
from __future__ import annotations

import re

MIN_WORDS_PER_SECOND = 0.5
MAX_WORDS_PER_SECOND = 5.0
REPETITION_NGRAM = 3
REPETITION_MIN_REPEATS = 3

# Column names we'll look for in a row's metadata dict, in priority order.
DURATION_KEYS = ["audio_duration", "audio_duration_ms", "duration", "duration_seconds"]
LANG_REQUESTED_KEYS = ["language_requested", "language"]
LANG_DETECTED_KEYS = ["language_detected"]


def _first_present(metadata: dict, keys: list[str]):
    for key in keys:
        if key in metadata and metadata[key] not in (None, "", "nan"):
            return metadata[key]
    return None


def _empty_output(hypothesis: str) -> bool:
    return not hypothesis or not hypothesis.strip()


def _repetition_detected(hypothesis: str) -> bool:
    words = hypothesis.lower().split()
    n = REPETITION_NGRAM
    if len(words) < n * REPETITION_MIN_REPEATS:
        return False
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        repeats = 1
        j = i + n
        while j + n <= len(words) and tuple(words[j:j + n]) == ngram:
            repeats += 1
            j += n
        if repeats >= REPETITION_MIN_REPEATS:
            return True
    return False


def _duration_seconds(raw_value, key: str) -> float | None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value / 1000.0 if key.endswith("_ms") else value


def _duration_anomaly(hypothesis: str, metadata: dict) -> bool | None:
    for key in DURATION_KEYS:
        if key in metadata and metadata[key] not in (None, "", "nan"):
            duration = _duration_seconds(metadata[key], key)
            if duration and duration > 0:
                word_count = len(hypothesis.split())
                wps = word_count / duration
                return wps < MIN_WORDS_PER_SECOND or wps > MAX_WORDS_PER_SECOND
    return None


def _language_mismatch(metadata: dict) -> bool | None:
    requested = _first_present(metadata, LANG_REQUESTED_KEYS)
    detected = _first_present(metadata, LANG_DETECTED_KEYS)
    if requested is None or detected is None:
        return None
    return str(requested).strip().lower() != str(detected).strip().lower()


def compute_quality_signals(hypothesis: str, metadata: dict) -> dict:
    """Returns a dict of signal_name -> True/False, omitting signals whose
    required input wasn't available in this row's metadata."""
    hypothesis = hypothesis or ""
    signals = {
        "empty_output": _empty_output(hypothesis),
        "repetition_detected": _repetition_detected(hypothesis),
    }

    duration_result = _duration_anomaly(hypothesis, metadata)
    if duration_result is not None:
        signals["duration_anomaly"] = duration_result

    language_result = _language_mismatch(metadata)
    if language_result is not None:
        signals["possible_language_mismatch"] = language_result

    return signals
