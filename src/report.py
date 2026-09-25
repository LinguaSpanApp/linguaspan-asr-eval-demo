"""Aggregate + per-slice reporting from a run_evaluation() results DataFrame.

Slice columns come from whatever metadata the input CSV happened to carry
(language, domain, audio condition, ...) plus model_name/model_version if the
API returned them -- this is the doc's "Sliceability" principle (section 5)
and what Error Intelligence (section 13) needs, without a live dashboard.
"""
from __future__ import annotations

import json
import pandas as pd

from . import confusions

MAX_SLICE_CARDINALITY = 50
SIGNAL_PREFIX = "signal."
DEFAULT_MIN_SLICE_SAMPLES = 30


def _corpus_wer(df: pd.DataFrame) -> float | None:
    scored = df.dropna(subset=["wer"])
    if scored.empty:
        return None
    errors = (scored["substitutions"] + scored["deletions"] + scored["insertions"]).sum()
    ref_words = (scored["substitutions"] + scored["deletions"] + scored["hits"]).sum()
    return errors / ref_words if ref_words else None


def _signal_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SIGNAL_PREFIX)]


def aggregate_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    with_reference = int(df["wer"].notna().sum()) if "wer" in df.columns else 0
    api_errors = int(df["api_error"].notna().sum()) if "api_error" in df.columns else 0

    summary = {
        "total_samples": total,
        "samples_with_reference": with_reference,
        "samples_without_reference": total - with_reference,
        "api_errors": api_errors,
    }

    if with_reference:
        scored = df.dropna(subset=["wer"])
        summary["corpus_wer"] = _corpus_wer(df)
        summary["mean_sample_wer"] = float(scored["wer"].mean())
        summary["mean_cer"] = float(scored["cer"].dropna().mean()) if scored["cer"].notna().any() else None
        summary["total_substitutions"] = int(scored["substitutions"].sum())
        summary["total_deletions"] = int(scored["deletions"].sum())
        summary["total_insertions"] = int(scored["insertions"].sum())

    for col in _signal_columns(df):
        applicable = df[col].notna()
        if applicable.any():
            summary[f"{col}_rate"] = float(df.loc[applicable, col].mean())
            summary[f"{col}_count"] = int(df.loc[applicable, col].sum())

    return summary


def slice_reports(df: pd.DataFrame, slice_columns: list[str]) -> dict[str, pd.DataFrame]:
    reports = {}
    signal_cols = _signal_columns(df)
    for col in slice_columns:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) > MAX_SLICE_CARDINALITY:
            continue

        agg = {"row_id": "count"}
        if "wer" in df.columns:
            agg["wer"] = "mean"
        if "cer" in df.columns:
            agg["cer"] = "mean"
        for signal_col in signal_cols:
            agg[signal_col] = "mean"

        grouped = df.groupby(col, dropna=False).agg(agg).rename(columns={"row_id": "sample_count"})
        reports[col] = grouped.reset_index()
    return reports


def coverage_gaps(
    df: pd.DataFrame, slice_columns: list[str], min_samples: int = DEFAULT_MIN_SLICE_SAMPLES,
) -> pd.DataFrame:
    """Doc section 13/21: "coverage gaps where a language, accent, domain, or
    acoustic condition has insufficient evaluation data." Flags slice VALUES
    (e.g. language='french') with fewer than min_samples evaluated rows --
    a thin slice's WER/CER number isn't trustworthy even if it looks fine."""
    gaps = []
    for col in slice_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            if count < min_samples:
                gaps.append({
                    "slice_column": col,
                    "slice_value": value,
                    "sample_count": int(count),
                    "min_required": min_samples,
                })
    return pd.DataFrame(gaps, columns=["slice_column", "slice_value", "sample_count", "min_required"])


def write_report(
    df: pd.DataFrame,
    slice_columns: list[str],
    output_dir: str,
    min_slice_samples: int = DEFAULT_MIN_SLICE_SAMPLES,
    confusion_top_n: int = 50,
) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    summary = aggregate_summary(df)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    for slice_name, slice_df in slice_reports(df, slice_columns).items():
        safe_name = slice_name.replace("/", "_")
        slice_df.to_csv(os.path.join(output_dir, f"slice_{safe_name}.csv"), index=False)

    gaps_df = coverage_gaps(df, slice_columns, min_samples=min_slice_samples)
    gaps_df.to_csv(os.path.join(output_dir, "coverage_gaps.csv"), index=False)

    for name, table in confusions.build_confusion_tables(df, top_n=confusion_top_n).items():
        table.to_csv(os.path.join(output_dir, f"confusions_{name}.csv"), index=False)

    print(json.dumps(summary, indent=2, default=str))
    if not gaps_df.empty:
        print(f"\n{len(gaps_df)} slice value(s) below the {min_slice_samples}-sample coverage threshold:")
        print(gaps_df.to_string(index=False))
    print(f"\nFull results + slice/confusion/coverage reports written to {output_dir}/")
