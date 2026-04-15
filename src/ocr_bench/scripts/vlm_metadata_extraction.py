"""Generic VLM metadata extraction runner for HF Jobs.

This script is used by humatheque-vlm-bench to run a specific model_id and
write outputs + inference_info to a Hub dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

try:
    from datasets import load_dataset
    from huggingface_hub import InferenceClient
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"])
    from datasets import load_dataset
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_PROMPT = """Extract the document title from this cover page.
Output ONLY valid JSON:
{
  "title": "Main title of the thesis as it appears on the title page",
  "subtitle": "Subtitle or remainder of the title, usually following a colon; null if not present",
  "author": "Full name of the author (student) who wrote the thesis",
  "degree_type": "Academic degree sought by the author.",
  "discipline": "Academic field or discipline of the thesis.",
  "granting_institution": "Institution where the thesis was submitted and the degree is granted",
  "doctoral_school": "Doctoral school or graduate program, if explicitly mentioned",
  "defense_year": "Year the thesis was defended. Format yyyy",
  "thesis_advisor": "Main thesis advisor or supervisor",
  "jury_president": "President or chair of the thesis examination committee",
  "reviewers": "Reviewers or rapporteurs of the thesis. Use | as separator",
  "committee_members": "Other thesis committee or jury members. Use | as separator",
  "language": "Language in ISO 639-3 codes. Example: fre, eng, ita..."
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM metadata extraction on a Hub dataset.")
    parser.add_argument("input_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image-column", default="image_uri")
    parser.add_argument("--output-column", default="prediction")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def _extract_one(client: InferenceClient, model_id: str, image_url: str, prompt: str, max_tokens: int, temperature: float) -> str:
    response = client.chat_completion(  # type: ignore[no-untyped-call]
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    args = parse_args()
    token = args.hf_token or os.getenv("HF_TOKEN")
    client = InferenceClient(token=token)

    ds = load_dataset(args.input_dataset, split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    if args.image_column not in ds.column_names:
        raise ValueError(f"Image column '{args.image_column}' not found in dataset.")

    predictions: list[str] = []
    for row in ds:
        image_url = row.get(args.image_column)
        if not image_url:
            predictions.append("{}")
            continue
        try:
            pred = _extract_one(
                client,
                args.model_id,
                image_url,
                args.prompt,
                args.max_tokens,
                args.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("inference_failed", extra={"error": str(exc)})
            pred = "{}"
        predictions.append(pred)

    info = json.dumps(
        [
            {
                "model_id": args.model_id,
                "model_name": args.model_id.split("/")[-1],
                "column_name": args.output_column,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
        ]
    )

    ds = ds.add_column(args.output_column, predictions)
    ds = ds.add_column("inference_info", [info] * len(ds))
    ds.push_to_hub(args.output_dataset, split=args.split, private=args.private, token=token)
    logger.info("metadata_extraction_complete", extra={"rows": len(ds), "model_id": args.model_id})


if __name__ == "__main__":
    main()
