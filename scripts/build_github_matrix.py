import argparse
import json
import re
from urllib.parse import urlparse


def parse_dataset_url(url: str) -> tuple[str, str]:
    """
    Example:
    https://huggingface.co/datasets/ai4bharat/IndicVoices/viewer/hindi

    Returns:
    dataset_name = ai4bharat/IndicVoices
    dataset_config = hindi
    """
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]

    # Expected:
    # datasets/{org}/{dataset}/viewer/{config}
    if len(parts) >= 5 and parts[0] == "datasets" and "viewer" in parts:
        viewer_index = parts.index("viewer")

        if viewer_index < 3:
            raise ValueError(f"Invalid Hugging Face dataset viewer URL: {url}")

        dataset_name = f"{parts[1]}/{parts[2]}"
        dataset_config = parts[viewer_index + 1]

        return dataset_name, dataset_config

    raise ValueError(
        f"Unsupported URL format: {url}. "
        f"Expected format like https://huggingface.co/datasets/org/dataset/viewer/config"
    )


def safe_run_name(dataset_name: str, dataset_config: str, split: str) -> str:
    raw = f"{dataset_name}_{dataset_config}_{split}".lower()
    raw = raw.replace("/", "_")
    raw = re.sub(r"[^a-z0-9_\\-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw)
    return raw.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urls-json",
        required=True,
        help="JSON array of Hugging Face viewer URLs.",
    )

    parser.add_argument(
        "--splits",
        default="train,valid",
        help="Comma-separated splits. Example: train,valid or train,test",
    )

    args = parser.parse_args()

    urls = json.loads(args.urls_json)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    if not isinstance(urls, list):
        raise ValueError("--urls-json must be a JSON array of strings.")

    if len(urls) > 20:
        raise ValueError("You can pass at most 20 URLs.")

    include = []

    for url in urls:
        dataset_name, dataset_config = parse_dataset_url(url)

        for split in splits:
            include.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_config": dataset_config,
                    "dataset_split": split,
                    "run_name": safe_run_name(dataset_name, dataset_config, split),
                }
            )

    print(json.dumps({"include": include}, ensure_ascii=False))


if __name__ == "__main__":
    main()