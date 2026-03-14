#!/usr/bin/env python3
"""
run_audit.py

Purpose:
- Run audit prompts from a CSV file against OpenAI GPT and/or Anthropic Claude
- Correctly route:
    --models gpt       -> OpenAI only
    --models claude    -> Anthropic only
    --models gpt,claude -> both
- Save all raw responses to a timestamped CSV

Required environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY   (only needed if using claude)

Example usage:
    python -u run_audit.py --models gpt --limit 2 --runs 1
    python -u run_audit.py --models gpt --runs 3
    python -u run_audit.py --models claude --runs 3
    python -u run_audit.py --models gpt,claude --runs 3
"""

import os
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional

from openai import OpenAI
import anthropic


# =========================
# CONFIG
# =========================
DEFAULT_INPUT_CSV = "vignettes.csv"
DEFAULT_OUTPUT_DIR = "results"
RUNS_PER_VIGNETTE = 3
SLEEP_BETWEEN_CALLS = 0.5

DEFAULT_GPT_MODEL = "gpt-4o"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"


# =========================
# API CLIENTS
# =========================
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


# =========================
# MODEL CALLS
# =========================
def call_gpt(client: OpenAI, prompt: str, model_name: str = DEFAULT_GPT_MODEL) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


def call_claude(
    client: anthropic.Anthropic,
    prompt: str,
    model_name: str = DEFAULT_CLAUDE_MODEL
) -> str:
    response = client.messages.create(
        model=model_name,
        max_tokens=800,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    parts = []
    for block in response.content:
        if hasattr(block, "text") and block.text:
            parts.append(block.text)

    return "\n".join(parts).strip()


# =========================
# PROMPT BUILDING
# =========================
def build_prompt(vignette_text: str) -> str:
    return vignette_text.strip()


# =========================
# CSV LOADING
# =========================
def load_vignettes(csv_path: str):
    """
    Expected CSV columns for this project:
    vignette_id, condition_id, prompt
    """

    rows = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV appears empty.")

        required = ["vignette_id", "condition_id", "prompt"]

        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(
                    f"CSV must contain column '{col}'. Found columns: {reader.fieldnames}"
                )

        for row in reader:
            vignette_text = row["prompt"].strip()

            if not vignette_text:
                continue

            rows.append(
                {
                    "id": row["vignette_id"],
                    "condition": row["condition_id"],
                    "vignette": vignette_text,
                }
            )

    return rows

       
          


# =========================
# OUTPUT HELPERS
# =========================
def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_output_path(output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"raw_responses_{timestamp}.csv")


def append_result(output_csv: str, row: Dict[str, str], write_header: bool = False) -> None:
    fieldnames = [
        "timestamp",
        "model",
        "run_idx",
        "id",
        "condition",
        "prompt",
        "response",
        "error",
    ]

    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =========================
# MODEL PARSING
# =========================
def parse_models(models_arg: str) -> List[str]:
    models = [m.strip().lower() for m in models_arg.split(",") if m.strip()]
    valid_models = {"gpt", "claude"}

    if not models:
        raise ValueError("No models provided. Use --models gpt, claude, or gpt,claude")

    for model in models:
        if model not in valid_models:
            raise ValueError(
                f"Unsupported model '{model}'. Use only: gpt, claude"
            )

    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)

    return deduped


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run audit prompts against GPT and/or Claude.")
    parser.add_argument(
        "--input_csv",
        default=DEFAULT_INPUT_CSV,
        help="Path to input vignette CSV",
    )
    parser.add_argument(
        "--models",
        default="gpt",
        help="Comma-separated models to run: gpt, claude, or gpt,claude",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_VIGNETTE,
        help="Runs per vignette",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV",
    )
    parser.add_argument(
        "--gpt_model",
        default=DEFAULT_GPT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_GPT_MODEL})",
    )
    parser.add_argument(
        "--claude_model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Anthropic model name (default: {DEFAULT_CLAUDE_MODEL})",
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 if provided")

    models = parse_models(args.models)

    ensure_output_dir(args.output_dir)
    output_csv = make_output_path(args.output_dir)

    vignettes = load_vignettes(args.input_csv)
    if args.limit is not None:
        vignettes = vignettes[:args.limit]

    if not vignettes:
        raise ValueError(f"No usable vignettes found in {args.input_csv}")

    # Create only the clients actually needed
    openai_client: Optional[OpenAI] = None
    anthropic_client: Optional[anthropic.Anthropic] = None

    if "gpt" in models:
        openai_client = get_openai_client()

    if "claude" in models:
        anthropic_client = get_anthropic_client()

    header_written = False

    for model_name in models:
        print(f"\n=== Running model: {model_name} ===")

        for item in vignettes:
            prompt = build_prompt(item["vignette"])

            for run_idx in range(1, args.runs + 1):
                timestamp = datetime.now().isoformat(timespec="seconds")

                try:
                    if model_name == "gpt":
                        if openai_client is None:
                            raise RuntimeError("OpenAI client was not initialized")
                        response = call_gpt(
                            openai_client,
                            prompt,
                            model_name=args.gpt_model,
                        )

                    elif model_name == "claude":
                        if anthropic_client is None:
                            raise RuntimeError("Anthropic client was not initialized")
                        response = call_claude(
                            anthropic_client,
                            prompt,
                            model_name=args.claude_model,
                        )

                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    error_text = ""
                    print(f"[OK] model={model_name} id={item['id']} run={run_idx}")

                except Exception as e:
                    response = ""
                    error_text = str(e)
                    print(f"[ERROR] model={model_name} id={item['id']} run={run_idx} -> {error_text}")

                append_result(
                    output_csv=output_csv,
                    row={
                        "timestamp": timestamp,
                        "model": model_name,
                        "run_idx": run_idx,
                        "id": item["id"],
                        "condition": item["condition"],
                        "prompt": prompt,
                        "response": response,
                        "error": error_text,
                    },
                    write_header=not header_written,
                )
                header_written = True

                time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()