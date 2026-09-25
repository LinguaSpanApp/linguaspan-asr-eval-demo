"""Orchestrates one evaluation run: dataset -> API calls -> metrics/signals -> rows.

Works whether or not the dataset has a reference column:
  - reference present  -> wer/cer/substitutions/deletions/insertions columns are filled
  - reference absent   -> those columns are left null; quality_signals columns still fill in
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pandas as pd

from . import metrics
from . import quality_signals
from .api_client import ASRClient, normalize_language
from .dataset import DatasetSchema

DEFAULT_CONCURRENCY = 4
YORUBA_MODEL_VARIANTS = ["base", "lora"]


def _row_metadata(row: pd.Series, schema: DatasetSchema) -> dict:
    return {col: row[col] for col in schema.metadata_cols}


def _model_choices_for_row(call_params: dict, compare_yoruba_models: bool) -> list[str]:
    """Normally a row runs once, with whatever model_choice it resolves to.
    If the row's language is Yoruba and comparison mode is on, it runs twice
    (base + lora) so the two variants can be diffed in the report."""
    language = call_params.get("language")
    if language and compare_yoruba_models:
        try:
            if normalize_language(language) == "yoruba":
                return list(YORUBA_MODEL_VARIANTS)
        except ValueError:
            pass  # unrecognized language -- let the real API call surface that error
    return [call_params.get("model_choice", "base")]


def _evaluate_row(
    row_id, audio_ref: str, reference: str | None, row_metadata: dict,
    client: ASRClient, default_params: dict, model_choice: str,
) -> dict:
    call_params = {**default_params, **row_metadata, "model_choice": model_choice}
    if reference is not None:
        call_params["reference_text"] = reference  # only read by MockASRClient
    result = client.transcribe(audio_ref, **call_params)

    record = {
        "row_id": row_id,
        "audio_ref": audio_ref,
        "model_choice": model_choice,
        "hypothesis_text": result.hypothesis_text,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "api_error": result.error,
        **row_metadata,
    }

    if reference is not None and result.error is None:
        score = metrics.score_sample(reference, result.hypothesis_text)
        record.update({
            "reference_text": reference,
            "wer": score.wer,
            "cer": score.cer,
            "substitutions": score.substitutions,
            "deletions": score.deletions,
            "insertions": score.insertions,
            "hits": score.hits,
        })
    elif reference is not None:
        record["reference_text"] = reference

    if result.error is None:
        for signal_name, flagged in quality_signals.compute_quality_signals(
            result.hypothesis_text, row_metadata
        ).items():
            record[f"signal.{signal_name}"] = flagged

    return record


def run_evaluation(
    df: pd.DataFrame,
    schema: DatasetSchema,
    client: ASRClient,
    concurrency: int = DEFAULT_CONCURRENCY,
    default_params: dict | None = None,
    compare_yoruba_models: bool = True,
) -> pd.DataFrame:
    default_params = default_params or {}
    rows = []
    tasks = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for idx, row in df.iterrows():
            row_id = row[schema.id_col] if schema.id_col else idx
            audio_ref = row[schema.audio_col]
            reference = None
            if schema.has_reference:
                raw_ref = row[schema.reference_col]
                reference = None if (raw_ref is None or (isinstance(raw_ref, float) and math.isnan(raw_ref))) else str(raw_ref)
            row_metadata = _row_metadata(row, schema)
            call_params = {**default_params, **row_metadata}

            variants = _model_choices_for_row(call_params, compare_yoruba_models)
            for model_choice in variants:
                # row_id is stringified uniformly -- some rows expand into two (base/lora)
                # and mixing int/str row_ids in the same column breaks the sort below.
                variant_row_id = f"{row_id}::{model_choice}" if len(variants) > 1 else str(row_id)
                tasks.append(pool.submit(
                    _evaluate_row, variant_row_id, audio_ref, reference, row_metadata,
                    client, default_params, model_choice,
                ))

        for future in as_completed(tasks):
            rows.append(future.result())

    results = pd.DataFrame(rows)
    if "row_id" in results.columns:
        results = results.sort_values("row_id").reset_index(drop=True)
    return results
