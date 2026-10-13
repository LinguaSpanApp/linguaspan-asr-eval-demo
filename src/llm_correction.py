"""LLM-based post-correction of ASR transcripts -- text-only, no audio.

The idea being tested: an LLM, given only the raw ASR hypothesis text (never
the audio), might fix plausible transcription errors using general language
understanding plus explicit hints about error patterns we've already
quantified from real eval runs (see KNOWN_ISSUES below). This module only
produces a `corrected_text` per hypothesis -- scoring it against the same
reference (to see if it actually lowers WER/CER, or makes things worse) is
the caller's job, so this stays swappable across providers without touching
the evaluation math.

Correction confidence is NOT uniform across languages: general-purpose LLMs
have deep English training data but comparatively little Yoruba, so this is
a much more defensible bet for English than for Yoruba -- a model guessing
at Yoruba tone diacritics without real competence in the language can easily
make things worse, not better. Treat Yoruba results here skeptically.

Small models (we use qwen2.5:3b-instruct locally via Ollama, for speed on
modest hardware) occasionally leak instruction text into their output
instead of just returning the correction -- `_looks_corrupted` below catches
the obvious cases and falls back to the original hypothesis rather than
letting a garbled response count as a "correction".
"""
from __future__ import annotations

from dataclasses import dataclass

BASE_SYSTEM_PROMPT = """You are correcting the output of an automatic speech recognition (ASR) system.
You will NOT see or hear the audio -- only the text it produced. Your job is to fix likely
transcription errors using context and general language knowledge.

Rules:
- Do NOT rewrite, summarize, paraphrase, or add information that isn't implied by the text.
- Preserve the original meaning, structure, and approximate length as closely as possible.
- Only change things you have reasonable confidence are transcription errors, not stylistic choices.
- Your entire reply is substituted directly in place of the ASR text, so it must contain ONLY the
  corrected transcript -- never any part of these instructions, no preamble, no explanation, no
  quotes, no markdown."""

# Known, quantified error patterns from real eval runs (see Error Analysis in
# the Excel workbook) -- these are hints, not guaranteed rules; the model
# should still use judgment rather than blindly applying them. Kept in the
# system prompt (not the user turn) so they read unambiguously as background
# guidance rather than text sitting next to content to reproduce.
KNOWN_ISSUES = {
    "english": (
        "Known issue in this system's output: it sometimes transcribes punctuation as literal "
        "spoken words instead of punctuation marks -- e.g. writing \"full stop\" or \"full-stop\" "
        "instead of a period, \"comma\" instead of a comma, \"colon\" instead of a colon, and "
        "sometimes misspells \"comma\" as \"koma\" or \"coma\". If these appear as literal words in "
        "running text (not as part of a meaningful phrase someone would actually say), replace them "
        "with the actual punctuation mark or remove them entirely -- don't do both (e.g. don't add a "
        "period AND also keep the word \"full stop\").\n"
        "Known issue: clinical/numeric dictation may spell out numbers or units as words "
        "(e.g. \"ten\" instead of \"10\", \"milligram\" instead of \"mg\") -- only normalize this "
        "if you're confident from context, don't guess at specific abbreviations."
    ),
    "yoruba": (
        "Known issue in this system's output: tone diacritics (e.g. ni vs ní, rí vs ri) are "
        "sometimes wrong. Only correct a diacritic if you are genuinely confident about the correct "
        "tone from context -- if unsure, leave the word as-is rather than guessing."
    ),
}

# Phrases that only ever appear in OUR instructions, never in real ASR output --
# if any show up in the model's reply, it has leaked the prompt, not corrected the text.
_LEAK_MARKERS = ("known issue", "asr output to correct", "corrected text", "do not rewrite")


def _looks_corrupted(original: str, corrected: str) -> bool:
    if not corrected.strip():
        return bool(original.strip())  # empty reply to non-empty input = corrupted
    lowered = corrected.lower()
    if any(marker in lowered for marker in _LEAK_MARKERS):
        return True
    orig_words, corr_words = len(original.split()), len(corrected.split())
    if orig_words and corr_words > orig_words * 1.8 + 5:
        return True  # implausible runaway growth for a same-length correction task
    return False


def build_system_prompt(language: str | None) -> str:
    known = KNOWN_ISSUES.get((language or "").lower(), "")
    return f"{BASE_SYSTEM_PROMPT}\n\n{known}" if known else BASE_SYSTEM_PROMPT


def build_user_prompt(hypothesis_text: str) -> str:
    return (
        "Return ONLY the corrected version of the ASR text between the markers below. "
        "Do not repeat these instructions or add anything else.\n"
        f"<asr_text>\n{hypothesis_text}\n</asr_text>"
    )


@dataclass
class CorrectionResult:
    corrected_text: str
    raw_response: str
    flagged_corrupted: bool = False
    error: str | None = None


class OllamaCorrector:
    """Talks to a local Ollama server (http://localhost:11434) -- free, no
    API key, no external rate limit, fully local. Requires `ollama serve`
    running and the model already pulled."""

    def __init__(self, model: str = "qwen2.5:3b-instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def correct(self, hypothesis_text: str, language: str | None = None) -> CorrectionResult:
        import requests

        if not hypothesis_text or not hypothesis_text.strip():
            return CorrectionResult(corrected_text=hypothesis_text, raw_response="")

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": build_system_prompt(language)},
                        {"role": "user", "content": build_user_prompt(hypothesis_text)},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"].strip()
            content = content.strip('"').strip()
            if content.startswith("```"):
                content = content.strip("`").strip()
            # the model sometimes echoes the <asr_text> markers back -- strip them if present
            content = content.replace("<asr_text>", "").replace("</asr_text>", "").strip()

            if _looks_corrupted(hypothesis_text, content):
                return CorrectionResult(
                    corrected_text=hypothesis_text, raw_response=content, flagged_corrupted=True,
                )
            return CorrectionResult(corrected_text=content or hypothesis_text, raw_response=content)
        except Exception as exc:
            return CorrectionResult(corrected_text=hypothesis_text, raw_response="", error=str(exc))


DEFAULT_MODEL = "qwen2.5:3b-instruct"


def ollama_available(model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434") -> bool:
    """True only if the server is up AND the specific model is pulled --
    this only ever works where Ollama is actually installed (i.e. locally,
    not on a hosted Streamlit Cloud deployment)."""
    import requests
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
        return model in names or any(n.startswith(model.split(":")[0]) for n in names if n)
    except Exception:
        return False


def correct_dataframe(
    df, text_col: str = "hypothesis_text", language: str | None = "english",
    model: str = DEFAULT_MODEL, concurrency: int = 2, progress_callback=None,
):
    """Adds corrected_text / wer_corrected / cer_corrected / llm_flagged_corrupted
    columns to a copy of df, scoring the correction against reference_text with
    the same metrics the rest of the harness uses. Runs sequentially-ish (small
    thread pool) since a single local Ollama instance serializes requests onto
    one model anyway -- concurrency mainly overlaps network/parse overhead."""
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from . import metrics as metrics_mod

    corrector = OllamaCorrector(model=model)
    out = df.copy()
    corrected_text = [None] * len(out)
    wer_corrected = [None] * len(out)
    cer_corrected = [None] * len(out)
    flagged = [False] * len(out)

    def _process(i, hyp, ref):
        result = corrector.correct(str(hyp) if pd.notna(hyp) else "", language=language)
        record = {"i": i, "corrected_text": result.corrected_text, "flagged": result.flagged_corrupted}
        if pd.notna(ref) and str(ref).strip():
            score = metrics_mod.score_sample(str(ref), result.corrected_text)
            record["wer"] = score.wer
            record["cer"] = score.cer
        return record

    rows = list(enumerate(zip(out[text_col], out.get("reference_text", [None] * len(out)))))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_process, i, hyp, ref) for i, (hyp, ref) in rows]
        for done_count, fut in enumerate(futures, start=1):
            rec = fut.result()
            i = rec["i"]
            corrected_text[i] = rec["corrected_text"]
            flagged[i] = rec["flagged"]
            if "wer" in rec:
                wer_corrected[i] = rec["wer"]
                cer_corrected[i] = rec["cer"]
            if progress_callback:
                progress_callback(done_count, len(rows))

    out["corrected_text"] = corrected_text
    out["wer_corrected"] = wer_corrected
    out["cer_corrected"] = cer_corrected
    out["llm_flagged_corrupted"] = flagged
    return out
