"""Word-confusion analysis -- design doc section 13's "most frequent
substitutions and deletions", the human-readable companion to the raw
substitution/deletion/insertion COUNTS metrics.py already produces.

Built directly from a results DataFrame's reference_text/hypothesis_text
columns (re-running jiwer's alignment, which is cheap pure-text work), so it
works on any completed run's results.csv without touching evaluate.py.
"""
from __future__ import annotations

from collections import Counter
import jiwer
import pandas as pd

from .metrics import _NORMALIZE


def _row_confusions(reference: str, hypothesis: str) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    output = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE,
    )
    ref_words = output.references[0]
    hyp_words = output.hypotheses[0]

    substitution_pairs: list[tuple[str, str]] = []
    deleted_words: list[str] = []
    inserted_words: list[str] = []

    for chunk in output.alignments[0]:
        ref_span = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_span = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type == "substitute":
            # jiwer groups consecutive substitutions into one chunk that can cover
            # unequal-length spans -- pair word-by-word, spill any excess into
            # deletions/insertions (matches how those extra words were actually scored).
            substitution_pairs.extend(zip(ref_span, hyp_span))
            if len(ref_span) > len(hyp_span):
                deleted_words.extend(ref_span[len(hyp_span):])
            elif len(hyp_span) > len(ref_span):
                inserted_words.extend(hyp_span[len(ref_span):])
        elif chunk.type == "delete":
            deleted_words.extend(ref_span)
        elif chunk.type == "insert":
            inserted_words.extend(hyp_span)

    return substitution_pairs, deleted_words, inserted_words


def build_confusion_tables(df: pd.DataFrame, top_n: int = 50) -> dict[str, pd.DataFrame]:
    """Returns {"substitutions": ..., "deletions": ..., "insertions": ...},
    each a DataFrame of the top_n most frequent confusions across the whole run."""
    required = {"wer", "reference_text", "hypothesis_text"}
    if not required.issubset(df.columns):
        return {}

    scored = df.dropna(subset=list(required))
    substitution_counter: Counter[tuple[str, str]] = Counter()
    deletion_counter: Counter[str] = Counter()
    insertion_counter: Counter[str] = Counter()

    for _, row in scored.iterrows():
        subs, dels, ins = _row_confusions(str(row["reference_text"]), str(row["hypothesis_text"]))
        substitution_counter.update(subs)
        deletion_counter.update(dels)
        insertion_counter.update(ins)

    substitutions_df = pd.DataFrame(
        [(ref, hyp, count) for (ref, hyp), count in substitution_counter.most_common(top_n)],
        columns=["reference_word", "hypothesis_word", "count"],
    )
    deletions_df = pd.DataFrame(
        deletion_counter.most_common(top_n), columns=["word", "count"],
    )
    insertions_df = pd.DataFrame(
        insertion_counter.most_common(top_n), columns=["word", "count"],
    )

    return {
        "substitutions": substitutions_df,
        "deletions": deletions_df,
        "insertions": insertions_df,
    }
