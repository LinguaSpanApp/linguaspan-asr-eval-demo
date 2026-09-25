"""Generic Hugging Face dataset adapter -- given ANY HF dataset repo (not just
the one naijavoices set we smoke-tested with), sample N rows and produce a
DataFrame shaped like what dataset.py/evaluate.py already expect (an
audio_path column + a reference_text column + whatever metadata columns the
dataset happens to carry), so the rest of the harness doesn't change at all.

Uses `datasets` in streaming mode so we never download more than the N
sampled rows, regardless of how large the underlying dataset is.
"""
from __future__ import annotations

import os
import tempfile

TEXT_COLUMN_CANDIDATES = [
    "text", "transcription", "transcript", "sentence", "reference", "reference_text", "label",
]


def _patch_legacy_cache_check() -> None:
    """datasets.DatasetBuilder._check_legacy_cache2 dill-hashes config.data_files
    purely to look for a pre-3.0 cache directory layout -- optional, and on some
    Python builds (seen on Streamlit Cloud's Python 3.14) the dill pickling
    inside it raises a TypeError unrelated to anything about the actual
    dataset. We never rely on an old-format cache dir, so skipping this
    check entirely is safe regardless of Python version."""
    try:
        from datasets.builder import DatasetBuilder
        DatasetBuilder._check_legacy_cache2 = lambda self, dataset_module: None
    except Exception:
        pass


_patch_legacy_cache_check()


def parse_hf_repo_id(url_or_id: str, repo_type: str = "dataset") -> str:
    """Accepts a bare 'org/name' or a full huggingface.co URL (with optional
    /tree/main, /blob/main/..., etc. suffix) and returns 'org/name'."""
    value = url_or_id.strip().rstrip("/")
    if value.startswith("http://") or value.startswith("https://"):
        value = value.split("huggingface.co/", 1)[-1]
        parts = value.split("/")
        if repo_type == "dataset" and parts and parts[0] == "datasets":
            parts = parts[1:]
        value = "/".join(parts[:2])
    return value.strip("/")


def list_configs_and_splits(repo_id: str) -> dict[str, list[str]]:
    """{config_name: [split_names]}. A dataset with no distinct configs
    reports its single config as "default"."""
    from datasets import get_dataset_config_names, get_dataset_split_names

    token = os.environ.get("HF_TOKEN")
    configs = get_dataset_config_names(repo_id, token=token)
    result = {}
    for config in configs:
        try:
            result[config] = get_dataset_split_names(repo_id, config_name=config, token=token)
        except Exception:
            result[config] = ["train"]
    return result


def _detect_columns(features) -> tuple[str, str]:
    audio_col = None
    for name, feature in features.items():
        if type(feature).__name__ == "Audio":
            audio_col = name
            break
    if audio_col is None:
        raise ValueError(f"No audio-typed column found in this dataset's features: {list(features)}")

    lower_map = {name.lower(): name for name in features}
    text_col = None
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_map:
            text_col = lower_map[candidate]
            break
    if text_col is None:
        for name, feature in features.items():
            if name == audio_col:
                continue
            if type(feature).__name__ == "Value" and getattr(feature, "dtype", "") == "string":
                text_col = name
                break
    if text_col is None:
        raise ValueError(f"No text/reference column found in this dataset's features: {list(features)}")

    return audio_col, text_col


def sample_dataset(
    repo_id: str, n: int = 30, config: str | None = None, split: str = "train",
    audio_dir: str | None = None,
):
    """Returns (DataFrame, info dict). DataFrame has audio_path, reference_text,
    plus any other scalar columns as metadata -- same shape run_evaluation()
    already expects from a CSV via dataset.py."""
    import pandas as pd
    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(repo_id, name=config, split=split, streaming=True, token=token)
    audio_col, text_col = _detect_columns(ds.features)
    metadata_cols = [c for c in ds.features if c not in (audio_col, text_col)]

    # decode=False: we only want raw bytes (Modal decodes remotely), so skip
    # needing torchcodec/soundfile locally just to read this dataset's audio.
    ds = ds.cast_column(audio_col, Audio(decode=False))

    audio_dir = audio_dir or tempfile.mkdtemp(prefix="hf_audio_")
    rows = []
    for i, example in enumerate(ds.take(n)):
        audio = example[audio_col]
        audio_bytes = audio.get("bytes")
        source_path = audio.get("path") or ""
        if not audio_bytes:
            # rare: dataset stores only a path/URL with no embedded bytes
            import requests
            if source_path.startswith("http"):
                audio_bytes = requests.get(source_path, timeout=60).content
            else:
                with open(source_path, "rb") as f:
                    audio_bytes = f.read()

        ext = os.path.splitext(source_path)[1] or ".wav"
        audio_path = os.path.join(audio_dir, f"sample_{i}{ext}")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        row = {"audio_path": audio_path, "reference_text": example[text_col]}
        for col in metadata_cols:
            value = example[col]
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[col] = value
        rows.append(row)

    info = {
        "repo_id": repo_id, "config": config, "split": split,
        "audio_col": audio_col, "text_col": text_col, "audio_dir": audio_dir,
    }
    return pd.DataFrame(rows), info
