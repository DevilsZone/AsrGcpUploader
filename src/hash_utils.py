import hashlib


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transcript_hash(transcript: str) -> str:
    return sha256_text(normalize_text(transcript))


def record_hash(
    dataset_name: str,
    dataset_config: str,
    split: str,
    index: int,
    transcript: str,
) -> str:
    raw = "|".join(
        [
            dataset_name,
            dataset_config,
            split,
            str(index),
            normalize_text(transcript),
        ]
    )

    return sha256_text(raw)