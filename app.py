"""Streamlit UI: paste a Hugging Face model URL and a Hugging Face dataset
URL, click Evaluate, see the same reports run_eval.py produces (summary,
per-slice WER/CER, confusion tables, coverage gaps) rendered on the page.

Run with:
    streamlit run app.py

Needs `modal deploy modal_app.py` to have been run first (this UI calls
GenericASR, which loads whatever model repo you type in at call time).
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

from src import confusions, report  # noqa: E402
from src.dataset import detect_schema  # noqa: E402
from src.evaluate import run_evaluation  # noqa: E402
from src.generic_modal_client import GenericModalASRClient  # noqa: E402
from src.hf_dataset import list_configs_and_splits, parse_hf_repo_id, sample_dataset  # noqa: E402

st.set_page_config(page_title="Linguaspan ASR Eval", layout="wide")

SAMPLE_META_COLUMNS = ("language", "speaker_id", "gender", "age_range", "model_choice", "wer", "cer")
MAX_DRILLDOWN_SAMPLES = 25


def _download_csv_button(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


def _sample_label(row: pd.Series) -> str:
    text = row.get("reference_text")
    if pd.isna(text) or not text:
        text = row.get("hypothesis_text") or row.get("audio_ref", "")
    text = str(text)
    if len(text) > 60:
        text = text[:57] + "..."
    wer = row.get("wer")
    wer_part = f" (wer={wer:.2f})" if pd.notna(wer) else ""
    return f"{row['row_id']}: {text}{wer_part}"


def _render_sample(row: pd.Series) -> None:
    """One sample's detail + audio player -- used both for free browsing of
    the per-sample table and for confusion drill-down results."""
    cols = st.columns([3, 2])
    with cols[0]:
        if pd.notna(row.get("reference_text")):
            st.markdown(f"**Reference:** {row['reference_text']}")
        st.markdown(f"**Hypothesis:** {row.get('hypothesis_text', '')}")
        meta_bits = [
            f"{col}: {row[col]}" for col in SAMPLE_META_COLUMNS
            if col in row.index and pd.notna(row[col])
        ]
        if meta_bits:
            st.caption(" · ".join(meta_bits))
    with cols[1]:
        audio_ref = row.get("audio_ref")
        if audio_ref and (str(audio_ref).startswith("http") or os.path.exists(str(audio_ref))):
            st.audio(str(audio_ref))
        else:
            st.caption("Audio not available (local temp file may be gone, e.g. after a server restart).")
    st.divider()


def _confusion_options(table_df: pd.DataFrame, error_type: str) -> dict[str, tuple]:
    options: dict[str, tuple] = {}
    for _, r in table_df.iterrows():
        if error_type == "substitution":
            label = f"{r['reference_word']} → {r['hypothesis_word']}  ({r['count']}x)"
            key = ("substitution", r["reference_word"], r["hypothesis_word"])
        elif error_type == "deletion":
            label = f"\"{r['word']}\" deleted  ({r['count']}x)"
            key = ("deletion", r["word"], "")
        else:
            label = f"\"{r['word']}\" inserted  ({r['count']}x)"
            key = ("insertion", "", r["word"])
        options[label] = key
    return options


def _render_confusion_drilldown(
    table_df: pd.DataFrame, error_type: str, index: dict, results_df: pd.DataFrame,
) -> None:
    if table_df.empty:
        st.caption("No confusions of this type to drill into.")
        return

    options = _confusion_options(table_df, error_type)
    choice = st.selectbox(
        "Find audio samples where this happened:", options=list(options.keys()), key=f"drilldown_{error_type}",
    )
    if not choice:
        return

    row_ids = index.get(options[choice], [])
    if not row_ids:
        st.info("No matching samples found.")
        return

    matched = results_df[results_df["row_id"].isin(row_ids)]
    st.caption(f"{len(matched)} sample(s) contain this confusion:")
    for _, row in matched.head(MAX_DRILLDOWN_SAMPLES).iterrows():
        _render_sample(row)
    if len(matched) > MAX_DRILLDOWN_SAMPLES:
        st.caption(f"...and {len(matched) - MAX_DRILLDOWN_SAMPLES} more (showing first {MAX_DRILLDOWN_SAMPLES}).")


def _expected_password() -> str | None:
    try:
        value = st.secrets.get("APP_PASSWORD")
        if value:
            return value
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD")


def _require_password() -> bool:
    """No APP_PASSWORD configured -> open (local dev). Configured -> gate
    the whole page behind it (used for the temporary public demo deploy)."""
    expected = _expected_password()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("ASR Model Evaluation")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if password == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not _require_password():
    st.stop()

st.title("ASR Model Evaluation")
st.caption("Point at any Hugging Face model + dataset, run a batch evaluation via Modal, and review the report below.")

col1, col2 = st.columns(2)
with col1:
    model_input = st.text_input(
        "Hugging Face model URL or repo ID",
        placeholder="LinguaSpanApp/Yoruba-model-prod-finetunning-aug28",
    )
with col2:
    dataset_input = st.text_input(
        "Hugging Face dataset URL or repo ID",
        placeholder="naijavoices/naijavoices-dataset",
    )

config_name = None
split_name = "train"
if dataset_input:
    try:
        dataset_repo_preview = parse_hf_repo_id(dataset_input, repo_type="dataset")
        configs_splits = list_configs_and_splits(dataset_repo_preview)
        config_options = list(configs_splits.keys())
        col3, col4 = st.columns(2)
        with col3:
            config_name = st.selectbox("Dataset config", options=config_options)
        with col4:
            split_name = st.selectbox("Split", options=configs_splits.get(config_name, ["train"]))
    except Exception as exc:
        st.warning(f"Could not list configs/splits for this dataset yet: {exc}")

sample_size = st.number_input("Sample size", min_value=1, max_value=1000, value=30, step=1)
min_slice_samples = st.number_input(
    "Coverage-gap threshold (min samples per slice)", min_value=1, max_value=500, value=max(5, sample_size // 6),
)

run_clicked = st.button("Evaluate", type="primary")

if run_clicked:
    if not model_input or not dataset_input:
        st.error("Provide both a model and a dataset.")
        st.stop()

    model_repo = parse_hf_repo_id(model_input, repo_type="model")
    dataset_repo = parse_hf_repo_id(dataset_input, repo_type="dataset")

    with st.status("Running evaluation...", expanded=True) as status:
        st.write(f"Sampling {sample_size} rows from `{dataset_repo}` ({config_name}/{split_name})...")
        df, info = sample_dataset(dataset_repo, n=int(sample_size), config=config_name, split=split_name)
        st.write(f"Sampled {len(df)} rows. audio column: `{info['audio_col']}`, reference column: `{info['text_col']}`")

        schema = detect_schema(df)
        client = GenericModalASRClient(model_repo)

        st.write(f"Calling `{model_repo}` on Modal for each sample (first call may be slow: cold start + model load)...")
        # compare_yoruba_models=False: that base/lora expansion is meaningless here --
        # GenericModalASRClient always calls the one model_repo given, ignoring model_choice,
        # so "comparing" would just call the same model twice per row for nothing.
        results = run_evaluation(df, schema, client, concurrency=4, default_params={}, compare_yoruba_models=False)
        status.update(label="Evaluation complete", state="complete")

    st.session_state["results"] = results
    st.session_state["metadata_cols"] = schema.metadata_cols
    st.session_state["min_slice_samples"] = int(min_slice_samples)
    st.session_state["model_repo"] = model_repo
    st.session_state["dataset_repo"] = dataset_repo

if "results" in st.session_state:
    results: pd.DataFrame = st.session_state["results"]
    metadata_cols: list[str] = st.session_state["metadata_cols"]
    threshold: int = st.session_state["min_slice_samples"]

    st.header(f"Results: {st.session_state['model_repo']} on {st.session_state['dataset_repo']}")

    summary = report.aggregate_summary(results)
    cols = st.columns(5)
    cols[0].metric("Samples", summary.get("total_samples", 0))
    cols[1].metric("With reference", summary.get("samples_with_reference", 0))
    wer = summary.get("corpus_wer")
    cols[2].metric("Corpus WER", f"{wer:.3f}" if wer is not None else "n/a")
    cer = summary.get("mean_cer")
    cols[3].metric("Mean CER", f"{cer:.3f}" if cer is not None else "n/a")
    cols[4].metric("API errors", summary.get("api_errors", 0))
    st.download_button(
        "Download summary.json", json.dumps(summary, indent=2, default=str).encode("utf-8"),
        file_name="summary.json", mime="application/json", key="dl_summary",
    )

    st.subheader("Coverage gaps")
    gaps = report.coverage_gaps(results, metadata_cols, min_samples=threshold)
    if gaps.empty:
        st.success(f"No slice fell below the {threshold}-sample threshold.")
    else:
        st.warning(f"{len(gaps)} slice value(s) below the {threshold}-sample coverage threshold:")
        st.dataframe(gaps, use_container_width=True)
        _download_csv_button(gaps, "coverage_gaps.csv", "Download coverage_gaps.csv", key="dl_gaps")

    st.subheader("Slice breakdowns")
    slices = report.slice_reports(results, metadata_cols)
    if slices:
        tabs = st.tabs(list(slices.keys()))
        for tab, (name, slice_df) in zip(tabs, slices.items()):
            with tab:
                st.dataframe(slice_df, use_container_width=True)
                _download_csv_button(slice_df, f"slice_{name}.csv", f"Download slice_{name}.csv", key=f"dl_slice_{name}")
    else:
        st.info("No metadata columns to slice by.")

    st.subheader("Word confusions")
    st.caption("Pick a confusion below to pull up the actual audio samples it happened in.")
    tables = confusions.build_confusion_tables(results)
    confusion_index = confusions.build_confusion_index(results)
    tab_sub, tab_del, tab_ins = st.tabs(["Substitutions", "Deletions", "Insertions"])
    with tab_sub:
        sub_table = tables.get("substitutions", pd.DataFrame())
        st.dataframe(sub_table, use_container_width=True)
        _download_csv_button(sub_table, "confusions_substitutions.csv", "Download substitutions.csv", key="dl_sub")
        _render_confusion_drilldown(sub_table, "substitution", confusion_index, results)
    with tab_del:
        del_table = tables.get("deletions", pd.DataFrame())
        st.dataframe(del_table, use_container_width=True)
        _download_csv_button(del_table, "confusions_deletions.csv", "Download deletions.csv", key="dl_del")
        _render_confusion_drilldown(del_table, "deletion", confusion_index, results)
    with tab_ins:
        ins_table = tables.get("insertions", pd.DataFrame())
        st.dataframe(ins_table, use_container_width=True)
        _download_csv_button(ins_table, "confusions_insertions.csv", "Download insertions.csv", key="dl_ins")
        _render_confusion_drilldown(ins_table, "insertion", confusion_index, results)

    st.subheader("Per-sample results")
    st.dataframe(results, use_container_width=True)
    _download_csv_button(results, "results.csv", "Download results.csv", key="dl_results")

    if "audio_ref" in results.columns:
        st.subheader("Listen to a sample")
        options = {_sample_label(row): row["row_id"] for _, row in results.iterrows()}
        picked_label = st.selectbox("Pick a sample to review:", options=list(options.keys()), key="sample_picker")
        if picked_label:
            picked_row = results[results["row_id"] == options[picked_label]].iloc[0]
            _render_sample(picked_row)
