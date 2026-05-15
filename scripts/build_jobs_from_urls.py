import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def parse_dataset_url(url: str) -> tuple[str, str]:
    """
    Expected format:
    https://huggingface.co/datasets/ai4bharat/IndicVoices/viewer/hindi

    Returns:
    dataset_name   = ai4bharat/IndicVoices
    dataset_config = hindi
    """
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]

    if len(parts) >= 5 and parts[0] == "datasets" and "viewer" in parts:
        viewer_index = parts.index("viewer")

        dataset_name = f"{parts[1]}/{parts[2]}"
        dataset_config = parts[viewer_index + 1]

        return dataset_name, dataset_config

    raise ValueError(
        f"Unsupported URL format: {url}. "
        f"Expected: https://huggingface.co/datasets/org/dataset/viewer/config"
    )


def safe_run_name(dataset_name: str, dataset_config: str, split: str) -> str:
    raw = f"{dataset_name}_{dataset_config}_{split}".lower()
    raw = raw.replace("/", "_")
    raw = re.sub(r"[^a-z0-9_\\-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw)
    return raw.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--urls-json", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-file", required=True)

    args = parser.parse_args()

    urls = json.loads(args.urls_json)

    if not isinstance(urls, list):
        raise ValueError("--urls-json must be a JSON array.")

    if len(urls) > 20:
        raise ValueError("You can pass at most 20 URLs.")

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    if not splits:
        raise ValueError("At least one split is required.")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for url in urls:
            dataset_name, dataset_config = parse_dataset_url(url)

            for split in splits:
                row = {
                    "dataset_name": dataset_name,
                    "dataset_config": dataset_config,
                    "dataset_split": split,
                    "run_name": safe_run_name(dataset_name, dataset_config, split),
                    "source_url": url,
                }

                file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()