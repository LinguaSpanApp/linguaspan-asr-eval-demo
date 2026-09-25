"""Loads an evaluation CSV and figures out which columns are which.

Handles two shapes of input, decided per-file at load time:
  - with reference text (+ optional metadata) -> full WER/CER scoring
  - audio only, no reference -> quality-signal-only scoring (see quality_signals.py)

Metadata is whatever is left over after audio/reference/id columns are
claimed; those columns become slice keys in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

AUDIO_COLUMN_CANDIDATES = [
    "audio_path", "audio_url", "audio", "audio_file", "file", "filepath", "path",
]
REFERENCE_COLUMN_CANDIDATES = [
    "reference", "reference_text", "transcript", "transcription",
    "ground_truth", "gold_text", "text", "label",
]
ID_COLUMN_CANDIDATES = [
    "id", "sample_id", "utterance_id", "audio_id", "row_id",
]


@dataclass
class DatasetSchema:
    audio_col: str
    reference_col: str | None
    id_col: str | None
    metadata_cols: list[str] = field(default_factory=list)

    @property
    def has_reference(self) -> bool:
        return self.reference_col is not None


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    columns = list(df.columns)

    audio_col = _find_column(columns, AUDIO_COLUMN_CANDIDATES)
    if audio_col is None:
        raise ValueError(
            f"Could not find an audio column. Columns present: {columns}. "
            f"Expected one of: {AUDIO_COLUMN_CANDIDATES}"
        )

    reference_col = _find_column(columns, REFERENCE_COLUMN_CANDIDATES)
    id_col = _find_column(columns, ID_COLUMN_CANDIDATES)

    claimed = {audio_col, reference_col, id_col} - {None}
    metadata_cols = [c for c in columns if c not in claimed]

    return DatasetSchema(
        audio_col=audio_col,
        reference_col=reference_col,
        id_col=id_col,
        metadata_cols=metadata_cols,
    )


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, DatasetSchema]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    schema = detect_schema(df)
    return df, schema
