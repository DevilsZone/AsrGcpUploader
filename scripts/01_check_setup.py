import shutil
import subprocess

from google.cloud import storage

from src.config import load_config


def check_ffmpeg() -> None:
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and make sure it is available in PATH."
        )

    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg check failed: {result.stderr}")

    first_line = result.stdout.splitlines()[0]
    print(f"FFmpeg OK: {first_line}")


def check_gcp_bucket(bucket_name: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if not bucket.exists():
        raise RuntimeError(f"GCS bucket does not exist or is not accessible: {bucket_name}")

    print(f"GCS bucket OK: gs://{bucket_name}")


def check_hf_token(token: str | None) -> None:
    if token:
        print("HF_TOKEN found.")
    else:
        print(
            "HF_TOKEN not found in environment. "
            "If you used `huggingface-cli login`, loading may still work."
        )


def main() -> None:
    config = load_config()

    print("Checking setup...")
    check_hf_token(config.hf_token)
    check_ffmpeg()
    check_gcp_bucket(config.gcp_bucket_name)

    print("Setup check completed successfully.")


if __name__ == "__main__":
    main()