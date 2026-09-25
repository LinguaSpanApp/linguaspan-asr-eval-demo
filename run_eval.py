#!/usr/bin/env python3
"""CLI entrypoint: evaluate a linguaspanx ASR model against a CSV dataset.

Usage:
    python3 run_eval.py --csv data/my_dataset.csv --output reports/prod_run
    python3 run_eval.py --csv data/my_dataset.csv --output reports/modal_run --backend modal
    python3 run_eval.py --csv data/my_dataset.csv --output reports/dry_run --backend mock

See --backend below for what each option needs configured.
"""
import argparse
import os
import sys


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency): KEY=VALUE per line, '#' comments,
    tolerates spaces around '='. Existing environment variables are not overridden."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

from src.dataset import load_dataset
from src.evaluate import run_evaluation, DEFAULT_CONCURRENCY
from src.report import write_report, DEFAULT_MIN_SLICE_SAMPLES
from src.api_client import ASRClient, MockASRClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the evaluation dataset CSV")
    parser.add_argument("--output", required=True, help="Directory to write results/summary/slice reports into")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--backend", default="production", choices=["production", "modal", "mock"],
        help="'production' calls mlapi.linguaspanapp.com (needs LINGUASPAN_API_KEY); "
             "'modal' calls our own Modal deployment of the Yoruba model directly "
             "(needs `modal deploy modal_app.py` run first); 'mock' echoes reference "
             "text back for pipeline testing.",
    )
    parser.add_argument(
        "--language", default=None,
        help="Default language (yoruba|french|english, or an alias like 'yo') for rows "
             "with no per-row language column. Per-row values still win.",
    )
    parser.add_argument(
        "--model-choice", default="base", choices=["base", "lora"],
        help="Yoruba model variant for non-Yoruba/fallback cases (default: base). "
             "Ignored for Yoruba rows unless --no-compare-yoruba-models is set.",
    )
    parser.add_argument(
        "--compare-yoruba-models", action=argparse.BooleanOptionalAction, default=True,
        help="For every Yoruba row, call the API twice (base + lora) and report both "
             "(default: on). Use --no-compare-yoruba-models to run just one variant.",
    )
    parser.add_argument(
        "--min-slice-samples", type=int, default=DEFAULT_MIN_SLICE_SAMPLES,
        help=f"Flag a slice value (e.g. one language, one domain) as a coverage gap "
             f"if it has fewer than this many evaluated rows (default: {DEFAULT_MIN_SLICE_SAMPLES}).",
    )
    args = parser.parse_args()

    df, schema = load_dataset(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  audio column:     {schema.audio_col}")
    print(f"  reference column: {schema.reference_col or '(none -- quality-signal-only run)'}")
    print(f"  metadata columns: {schema.metadata_cols or '(none)'}")

    compare_yoruba_models = args.compare_yoruba_models
    if args.backend == "mock":
        client = MockASRClient()
    elif args.backend == "modal":
        from src.modal_client import ModalASRClient
        client = ModalASRClient()
        if compare_yoruba_models:
            # modal_app.py deploys exactly one Yoruba model, no base/lora split --
            # running the base/lora comparison against it would just call the same
            # model twice and misleadingly report two "variants" with identical output.
            print("Note: --backend modal serves a single Yoruba model; disabling --compare-yoruba-models.")
            compare_yoruba_models = False
    else:
        client = ASRClient()
    default_params = {"model_choice": args.model_choice}
    if args.language:
        default_params["language"] = args.language

    results = run_evaluation(
        df, schema, client, concurrency=args.concurrency,
        default_params=default_params, compare_yoruba_models=compare_yoruba_models,
    )
    write_report(
        results, schema.metadata_cols + ["model_choice", "model_name", "model_version"], args.output,
        min_slice_samples=args.min_slice_samples,
    )


if __name__ == "__main__":
    sys.exit(main())
