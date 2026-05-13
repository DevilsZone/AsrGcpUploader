import argparse
import tempfile
from pathlib import Path

from src.audio_utils import convert_to_mp3, probe_audio, write_audio_value_to_wav
from src.checkpoint_utils import load_checkpoint, save_checkpoint
from src.config import load_config
from src.gcs_utils import GcsClient
from src.hash_utils import record_hash, transcript_hash
from src.hf_dataset import detect_required_columns, load_streaming_dataset
from src.manifest_utils import append_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of new rows to process in this run. Use -1 for no limit.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional manual start index. If omitted, checkpoint next_index is used.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional GCS run prefix. Overrides GCP_RUN_NAME from .env.",
    )

    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Start from zero and overwrite previous checkpoint for this run.",
    )

    return parser.parse_args()


def process_one_row(
    *,
    row: dict,
    index: int,
    config,
    run_name: str,
    gcs: GcsClient,
    manifest_path: Path,
) -> None:
    detected = detect_required_columns(row)

    audio_col = detected["audio"]
    transcript_col = detected["transcript"]
    lang_col = detected["lang"]
    gender_col = detected["gender"]

    if audio_col is None:
        raise ValueError(f"Audio column not found. Available columns: {list(row.keys())}")

    if transcript_col is None:
        raise ValueError(f"Transcript column not found. Available columns: {list(row.keys())}")

    transcript = str(row[transcript_col])
    lang = str(row[lang_col]) if lang_col else config.dataset_config
    gender = str(row[gender_col]) if gender_col else "unknown"

    record_id = record_hash(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        index=index,
        transcript=transcript,
    )

    shard = record_id[:2]
    audio_blob_name = f"{run_name}/audio/{shard}/{record_id}.mp3"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_wav_path = temp_dir_path / f"{record_id}.wav"
        temp_mp3_path = temp_dir_path / f"{record_id}.mp3"

        source_audio_path = write_audio_value_to_wav(
            audio_value=row[audio_col],
            output_wav_path=temp_wav_path,
        )

        convert_to_mp3(
            input_audio_path=source_audio_path,
            output_mp3_path=temp_mp3_path,
            target_sample_rate=config.target_sample_rate,
            target_channels=config.target_channels,
            mp3_bitrate=config.mp3_bitrate,
        )

        converted_probe = probe_audio(temp_mp3_path)

        audio_uri = gcs.upload_file(
            local_path=temp_mp3_path,
            destination_blob_name=audio_blob_name,
            skip_if_exists=True,
        )

    uploaded_size = gcs.get_blob_size(audio_blob_name)

    if uploaded_size is None or uploaded_size <= 0:
        raise RuntimeError(f"Upload verification failed for {audio_blob_name}")

    manifest_row = {
        "record_id": record_id,
        "source_dataset": config.dataset_name,
        "source_config": config.dataset_config,
        "source_split": config.dataset_split,
        "source_index": index,
        "audio_uri": audio_uri,
        "transcript": transcript,
        "lang": lang,
        "gender": gender,
        "transcript_hash": transcript_hash(transcript),
        "audio_format": "mp3",
        "sample_rate": int(converted_probe["sample_rate"]),
        "channels": int(converted_probe["channels"]),
        "gcs_size_bytes": uploaded_size,
    }

    append_jsonl(manifest_path, manifest_row)


def main() -> None:
    args = parse_args()
    config = load_config()

    run_name = args.run_name or config.gcp_run_name

    local_run_dir = Path(config.local_output_dir) / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = local_run_dir / "checkpoint.json"
    manifest_path = local_run_dir / f"{run_name}.jsonl"
    failures_path = local_run_dir / f"{run_name}_failures.jsonl"

    if args.reset_checkpoint:
        checkpoint = {
            "next_index": 0,
            "processed_count": 0,
            "failed_count": 0,
        }
        save_checkpoint(checkpoint_path, checkpoint)

        if manifest_path.exists():
            manifest_path.unlink()

        if failures_path.exists():
            failures_path.unlink()
    else:
        checkpoint = load_checkpoint(checkpoint_path)

    start_index = args.start_index if args.start_index is not None else checkpoint["next_index"]
    limit = None if args.limit == -1 else args.limit

    print("=" * 80)
    print("Starting resumable upload")
    print(f"Run name: {run_name}")
    print(f"Bucket: gs://{config.gcp_bucket_name}")
    print(f"Start index: {start_index}")
    print(f"Limit: {'no limit' if limit is None else limit}")
    print(f"Local manifest: {manifest_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    gcs = GcsClient(config.gcp_bucket_name)

    dataset = load_streaming_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        token=config.hf_token,
    )

    processed_this_run = 0

    for index, row in enumerate(dataset):
        if index < start_index:
            continue

        if limit is not None and processed_this_run >= limit:
            break

        try:
            process_one_row(
                row=row,
                index=index,
                config=config,
                run_name=run_name,
                gcs=gcs,
                manifest_path=manifest_path,
            )

            processed_this_run += 1

            checkpoint["next_index"] = index + 1
            checkpoint["processed_count"] = checkpoint.get("processed_count", 0) + 1
            save_checkpoint(checkpoint_path, checkpoint)

            print(f"SUCCESS index={index}, processed_this_run={processed_this_run}")

        except Exception as exception:
            failure_row = {
                "source_index": index,
                "error": str(exception),
            }

            append_jsonl(failures_path, failure_row)

            checkpoint["next_index"] = index + 1
            checkpoint["failed_count"] = checkpoint.get("failed_count", 0) + 1
            save_checkpoint(checkpoint_path, checkpoint)

            print(f"FAILED index={index}: {exception}")

    manifest_blob_name = f"{run_name}/manifests/{run_name}.jsonl"

    manifest_uri = gcs.upload_file(
        local_path=manifest_path,
        destination_blob_name=manifest_blob_name,
        skip_if_exists=False,
    )

    print("=" * 80)
    print("Run finished.")
    print(f"Processed in this run: {processed_this_run}")
    print(f"Next index: {checkpoint['next_index']}")
    print(f"Manifest uploaded to: {manifest_uri}")

    if failures_path.exists():
        failures_blob_name = f"{run_name}/manifests/{run_name}_failures.jsonl"
        failures_uri = gcs.upload_file(
            local_path=failures_path,
            destination_blob_name=failures_blob_name,
            skip_if_exists=False,
        )
        print(f"Failures uploaded to: {failures_uri}")


if __name__ == "__main__":
    main()