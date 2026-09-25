"""WER/CER and Recognition-level error breakdown (substitution/deletion/insertion).

Only used when a reference transcript is available for a row. This is the
automatic part of the Reviewer Taxonomy (doc Appendix B, "Recognition" level) --
the Acoustic/Language/Vocabulary/System causes behind an error still need a
human reviewer or a heuristic signal (see quality_signals.py), not this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import jiwer

_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class SampleScore:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_word_count: int


def score_sample(reference: str, hypothesis: str) -> SampleScore:
    reference = reference or ""
    hypothesis = hypothesis or ""

    word_output = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE,
    )
    cer = jiwer.cer(reference, hypothesis) if reference.strip() else float("nan")

    return SampleScore(
        wer=word_output.wer,
        cer=cer,
        substitutions=word_output.substitutions,
        deletions=word_output.deletions,
        insertions=word_output.insertions,
        hits=word_output.hits,
        ref_word_count=word_output.substitutions + word_output.deletions + word_output.hits,
    )
