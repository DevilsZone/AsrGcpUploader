import json
import sys
from itertools import islice
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.hf_dataset import load_streaming_dataset, detect_required_columns


def make_json_safe(value):
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "shape"):
        return f"<array shape={value.shape}>"

    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        if len(value) > 10:
            return [make_json_safe(item) for item in value[:10]] + ["..."]
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return tuple(make_json_safe(item) for item in value)

    return f"<non-json-serializable object: {type(value).__name__}>"


def main() -> None:
    config = load_config()

    dataset = load_streaming_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        token=config.hf_token,
    )

    print("Dataset stream created.")
    print("Reading first 3 rows...")

    for index, row in enumerate(islice(dataset, 3)):
        print("=" * 80)
        print(f"ROW INDEX: {index}")

        print("Columns:")
        print(list(row.keys()))

        detected = detect_required_columns(row)
        print("Detected columns:")
        print(json.dumps(detected, indent=2, ensure_ascii=False))

        audio_col = detected["audio"]
        transcript_col = detected["transcript"]
        lang_col = detected["lang"]
        gender_col = detected["gender"]

        print("Important values:")

        if audio_col:
            audio_value = row[audio_col]
            print(f"  audio column: {audio_col}")
            print(f"  audio value type: {type(audio_value).__name__}")
            print(f"  audio value preview: {make_json_safe(audio_value)}")
        else:
            print("  audio column: NOT FOUND")

        if transcript_col:
            print(f"  transcript column: {transcript_col}")
            print(f"  transcript: {row[transcript_col]}")

        if lang_col:
            print(f"  lang column: {lang_col}")
            print(f"  lang: {row[lang_col]}")

        if gender_col:
            print(f"  gender column: {gender_col}")
            print(f"  gender: {row[gender_col]}")

        safe_row = make_json_safe(row)

        print("Full safe row preview:")
        print(json.dumps(safe_row, indent=2, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()