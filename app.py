"""Streamlit UI: paste a Hugging Face model URL and a Hugging Face dataset
URL, click Evaluate, see the same reports run_eval.py produces (summary,
per-slice WER/CER, confusion tables, coverage gaps) rendered on the page.

Run with:
    streamlit run app.py

Needs `modal deploy modal_app.py` to have been run first (this UI calls
GenericASR, which loads whatever model repo you type in at call time).
"""
from __future__ import annotations

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

    st.subheader("Coverage gaps")
    gaps = report.coverage_gaps(results, metadata_cols, min_samples=threshold)
    if gaps.empty:
        st.success(f"No slice fell below the {threshold}-sample threshold.")
    else:
        st.warning(f"{len(gaps)} slice value(s) below the {threshold}-sample coverage threshold:")
        st.dataframe(gaps, use_container_width=True)

    st.subheader("Slice breakdowns")
    slices = report.slice_reports(results, metadata_cols)
    if slices:
        tabs = st.tabs(list(slices.keys()))
        for tab, (name, slice_df) in zip(tabs, slices.items()):
            with tab:
                st.dataframe(slice_df, use_container_width=True)
    else:
        st.info("No metadata columns to slice by.")

    st.subheader("Word confusions")
    tables = confusions.build_confusion_tables(results)
    tab_sub, tab_del, tab_ins = st.tabs(["Substitutions", "Deletions", "Insertions"])
    with tab_sub:
        st.dataframe(tables.get("substitutions", pd.DataFrame()), use_container_width=True)
    with tab_del:
        st.dataframe(tables.get("deletions", pd.DataFrame()), use_container_width=True)
    with tab_ins:
        st.dataframe(tables.get("insertions", pd.DataFrame()), use_container_width=True)

    st.subheader("Per-sample results")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download results.csv", results.to_csv(index=False).encode("utf-8"),
        file_name="results.csv", mime="text/csv",
    )
