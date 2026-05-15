from typing import Any

from datasets import Audio, load_dataset


def load_streaming_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    token: str | None,
) -> Any:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
        token=token,
    )

    # Important:

    # Avoid Hugging Face automatic audio decoding through TorchCodec.

    # This prevents Docker failures caused by torch/torchcodec/ffmpeg mismatch.

    try:

        dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

    except Exception as exception:

        print(

            f"Could not cast audio_filepath to Audio(decode=False). "

            f"Continuing without cast. Reason: {exception}"

        )

    return dataset


def find_first_available_column(row: dict, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return candidate

    return None


def detect_required_columns(row: dict) -> dict[str, str | None]:
    return {
        "audio": find_first_available_column(
            row,
            [
                "audio",
                "audio_filepath",
                "path",
                "wav",
                "file",
            ],
        ),
        "transcript": find_first_available_column(
            row,
            [
                "text",
                "transcript",
                "sentence",
                "normalized",
                "normalized_text",
                "raw_text",
                "verbatim",
            ],
        ),
        "lang": find_first_available_column(
            row,
            [
                "lang",
                "language",
                "language_code",
            ],
        ),
        "gender": find_first_available_column(
            row,
            [
                "gender",
                "speaker_gender",
                "sex",
            ],
        ),
    }