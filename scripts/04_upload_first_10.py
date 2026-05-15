from pathlib import Path

from src.config import load_config
from src.gcs_utils import GcsClient
from src.manifest_utils import append_jsonl, read_jsonl


def main() -> None:
    config = load_config()
    gcs = GcsClient(config.gcp_bucket_name)

    local_preview_manifest = (
        Path(config.local_output_dir)
        / "local_preview"
        / "preview_manifest.jsonl"
    )

    if not local_preview_manifest.exists():
        raise FileNotFoundError(
            f"Missing local preview manifest: {local_preview_manifest}. "
            f"Run scripts/03_convert_first_10.py first."
        )

    rows = read_jsonl(local_preview_manifest)

    upload_manifest_path = (
        Path(config.local_output_dir)
        / "local_preview"
        / "preview_10_gcs_manifest.jsonl"
    )

    if upload_manifest_path.exists():
        upload_manifest_path.unlink()

    for index, row in enumerate(rows):
        local_audio_path = Path(row["local_audio_path"])

        if not local_audio_path.exists():
            raise FileNotFoundError(f"Missing local audio file: {local_audio_path}")

        record_id = row["record_id"]
        shard = record_id[:2]

        audio_blob_name = f"{config.gcp_run_name}/audio/{shard}/{record_id}.wav"

        audio_uri = gcs.upload_file(
            local_path=local_audio_path,
            destination_blob_name=audio_blob_name,
            skip_if_exists=True,
        )

        uploaded_size = gcs.get_blob_size(audio_blob_name)

        if uploaded_size is None or uploaded_size <= 0:
            raise RuntimeError(f"Upload verification failed for {audio_blob_name}")

        output_row = {
            "record_id": row["record_id"],
            "source_dataset": row["source_dataset"],
            "source_config": row["source_config"],
            "source_split": row["source_split"],
            "source_index": row["source_index"],
            "audio_uri": audio_uri,
            "transcript": row["transcript"],
            "lang": row["lang"],
            "gender": row["gender"],
            "transcript_hash": row["transcript_hash"],
            "audio_format": "wav",
            "sample_rate": int(row["final_audio_probe"]["sample_rate"]),
            "channels": int(row["final_audio_probe"]["channels"]),
            "codec_name": row["final_audio_probe"]["codec_name"],
            "duration": row["final_audio_probe"].get("duration"),
            "gcs_size_bytes": uploaded_size,
        }

        append_jsonl(upload_manifest_path, output_row)

        print(f"Uploaded and verified {index + 1}/{len(rows)}")
        print(f"  {audio_uri}")

    manifest_blob_name = f"{config.gcp_run_name}/manifests/preview_10.jsonl"

    manifest_uri = gcs.upload_file(
        local_path=upload_manifest_path,
        destination_blob_name=manifest_blob_name,
        skip_if_exists=False,
    )

    manifest_size = gcs.get_blob_size(manifest_blob_name)

    if manifest_size is None or manifest_size <= 0:
        raise RuntimeError("Manifest upload verification failed.")

    print("=" * 80)
    print("Upload complete.")
    print(f"Manifest URI: {manifest_uri}")
    print(f"Manifest size: {manifest_size} bytes")


if __name__ == "__main__":
    main()