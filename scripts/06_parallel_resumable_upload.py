import argparse
import gc
import json
import os
import re
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from google.api_core import exceptions as gcp_exceptions
from google.api_core.retry import Retry, if_exception_type
from google.cloud import storage

from src.audio_utils import probe_audio, write_audio_value_to_wav
from src.checkpoint_utils import load_checkpoint, save_checkpoint
from src.config import load_config
from src.hash_utils import record_hash, transcript_hash
from src.hf_dataset import detect_required_columns, load_streaming_dataset


warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects.*",
    category=UserWarning,
)


_thread_local = threading.local()


class SimpleRateLimiter:
    def __init__(self, qps: float | None):
        self.qps = qps
        self.min_interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return

        with self.lock:
            now = time.monotonic()
            sleep_for = self.next_allowed_time - now

            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()

            self.next_allowed_time = now + self.min_interval

class BatchRateLimiter:
    def __init__(self, qps: float | None):
        self.qps = qps
        self.min_interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return

        with self.lock:
            now = time.monotonic()
            sleep_for = self.next_allowed_time - now

            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()

            self.next_allowed_time = now + self.min_interval


class FastGcsClient:
    def __init__(self, bucket_name: str, timeout_seconds: int):
        self.bucket_name = bucket_name
        self.timeout_seconds = timeout_seconds
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

        self.retry = Retry(
            predicate=if_exception_type(
                gcp_exceptions.TooManyRequests,
                gcp_exceptions.ServiceUnavailable,
                gcp_exceptions.InternalServerError,
                gcp_exceptions.BadGateway,
                gcp_exceptions.GatewayTimeout,
                gcp_exceptions.DeadlineExceeded,
            ),
            initial=1.0,
            maximum=30.0,
            multiplier=2.0,
            timeout=300.0,
        )

    def validate_bucket(self) -> None:
        if not self.bucket.exists(timeout=self.timeout_seconds, retry=self.retry):
            raise ValueError(
                f"GCS bucket does not exist or is not accessible: {self.bucket_name}"
            )

    def upload_file(
        self,
        *,
        local_path: Path,
        destination_blob_name: str,
        skip_if_exists: bool,
        rate_limiter: SimpleRateLimiter,
    ) -> tuple[str, int]:
        blob = self.bucket.blob(destination_blob_name)

        if skip_if_exists:
            if blob.exists(timeout=self.timeout_seconds, retry=self.retry):
                blob.reload(timeout=self.timeout_seconds, retry=self.retry)
                existing_size = int(blob.size or 0)

                if existing_size <= 0:
                    raise RuntimeError(
                        f"Existing blob has invalid size: {destination_blob_name}"
                    )

                return f"gs://{self.bucket_name}/{destination_blob_name}", existing_size

        rate_limiter.wait()

        blob.upload_from_filename(
            str(local_path),
            checksum="auto",
            timeout=self.timeout_seconds,
            retry=self.retry,
        )
        blob.reload(timeout=self.timeout_seconds, retry=self.retry)

        uploaded_size = int(blob.size or 0)

        if uploaded_size <= 0:
            raise RuntimeError(f"Uploaded blob has invalid size: {destination_blob_name}")

        return f"gs://{self.bucket_name}/{destination_blob_name}", uploaded_size


def get_thread_gcs_client(bucket_name: str, timeout_seconds: int) -> FastGcsClient:
    key = f"gcs_client_{bucket_name}_{timeout_seconds}"

    if not hasattr(_thread_local, key):
        setattr(_thread_local, key, FastGcsClient(bucket_name, timeout_seconds))

    return getattr(_thread_local, key)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config/upload_jobs.yaml",
        help="YAML config containing defaults and upload jobs.",
    )
    parser.add_argument(
        "--only-run-name",
        type=str,
        default=None,
        help="Run only one job by run_name.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Reset checkpoint for selected job/jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved jobs without uploading.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary.")

    return config


def infer_dataset_info_from_url(url: str) -> dict[str, str | None]:
    if not url:
        return {
            "dataset_name": None,
            "dataset_config": None,
            "split": None,
        }

    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]

    # Expected:
    # /datasets/ai4bharat/IndicVoices/viewer/bodo
    # /datasets/ai4bharat/IndicVoices/viewer/bodo/train
    if len(parts) < 5 or parts[0] != "datasets":
        return {
            "dataset_name": None,
            "dataset_config": None,
            "split": None,
        }

    owner = parts[1]
    dataset = parts[2]

    dataset_config = None
    split = None

    if "viewer" in parts:
        viewer_index = parts.index("viewer")

        if len(parts) > viewer_index + 1:
            dataset_config = parts[viewer_index + 1]

        if len(parts) > viewer_index + 2:
            split = parts[viewer_index + 2]

    return {
        "dataset_name": f"{owner}/{dataset}",
        "dataset_config": dataset_config,
        "split": split,
    }


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def resolve_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config.get("defaults", {})
    jobs = config.get("jobs", [])

    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a dictionary.")

    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list.")

    resolved_jobs = []

    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each job must be a dictionary.")

        merged = {**defaults, **job}

        inferred = infer_dataset_info_from_url(str(merged.get("url", "")))

        dataset_name = merged.get("dataset_name") or inferred["dataset_name"]
        dataset_config = merged.get("dataset_config") or inferred["dataset_config"]
        split = merged.get("split") or merged.get("dataset_split") or inferred["split"] or "valid"

        if not dataset_name:
            raise ValueError(f"Missing dataset_name for job: {job}")

        if not dataset_config:
            raise ValueError(f"Missing dataset_config for job: {job}")

        run_name = merged.get("run_name")

        if not run_name:
            run_name = slugify(
                f"{dataset_name.replace('/', '_')}_{dataset_config}_{split}"
            )

        resolved = {
            **merged,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": split,
            "run_name": run_name,
            "limit": int(merged.get("limit", -1)),
            "workers": int(merged.get("workers", 8)),
            "batch_size": int(merged.get("batch_size", 64)),
            "upload_qps": float(merged.get("upload_qps", 10)),
            "gcs_timeout_seconds": int(merged.get("gcs_timeout_seconds", 120)),
            "skip_if_exists": bool(merged.get("skip_if_exists", True)),
            "upload_state_to_gcs_every_batches": int(
                merged.get("upload_state_to_gcs_every_batches", 5)
            ),
            "local_output_dir": str(merged.get("local_output_dir", "output")),
            "job_parallelism": int(merged.get("job_parallelism", 1)),
            "query_batch_size": int(merged.get("query_batch_size", 20)),
            "query_qps": float(merged.get("query_qps", 0)),
            "max_prefetch_batches": int(merged.get("max_prefetch_batches", 2)),
        }

        resolved_jobs.append(resolved)

    return resolved_jobs


def append_jsonl_batch(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_state_files(
    *,
    gcs: FastGcsClient,
    run_name: str,
    manifest_path: Path,
    failures_path: Path,
    checkpoint_path: Path,
) -> None:
    no_limit = SimpleRateLimiter(None)

    if manifest_path.exists():
        gcs.upload_file(
            local_path=manifest_path,
            destination_blob_name=f"{run_name}/manifests/{manifest_path.name}",
            skip_if_exists=False,
            rate_limiter=no_limit,
        )

    if failures_path.exists():
        gcs.upload_file(
            local_path=failures_path,
            destination_blob_name=f"{run_name}/manifests/{failures_path.name}",
            skip_if_exists=False,
            rate_limiter=no_limit,
        )

    if checkpoint_path.exists():
        gcs.upload_file(
            local_path=checkpoint_path,
            destination_blob_name=f"{run_name}/state/checkpoint.json",
            skip_if_exists=False,
            rate_limiter=no_limit,
        )


def build_config_for_job(job: dict[str, Any]):
    base_config = load_config()

    return replace(
        base_config,
        hf_token=os.getenv("HF_TOKEN") or base_config.hf_token,
        gcp_bucket_name=os.getenv("GCP_BUCKET_NAME") or base_config.gcp_bucket_name,
        gcp_run_name=job["run_name"],
        dataset_name=job["dataset_name"],
        dataset_config=job["dataset_config"],
        dataset_split=job["dataset_split"],
        audio_format="wav",
        local_output_dir=job["local_output_dir"],
    )


def convert_hf_batch_to_rows(
    *,
    hf_batch: dict[str, list[Any]],
    start_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not hf_batch:
        return []

    first_key = next(iter(hf_batch))
    batch_size = len(hf_batch[first_key])

    rows = []

    for row_offset in range(batch_size):
        row = {
            column_name: column_values[row_offset]
            for column_name, column_values in hf_batch.items()
        }
        rows.append((start_index + row_offset, row))

    return rows


def iter_hf_batches(
    *,
    stream,
    start_index: int,
    query_batch_size: int,
    limit: int | None,
    query_qps: float,
):
    emitted = 0
    next_index = start_index
    limiter = BatchRateLimiter(query_qps if query_qps > 0 else None)

    for hf_batch in stream.iter(batch_size=query_batch_size):
        limiter.wait()

        rows = convert_hf_batch_to_rows(
            hf_batch=hf_batch,
            start_index=next_index,
        )

        if limit is not None:
            remaining = limit - emitted

            if remaining <= 0:
                break

            rows = rows[:remaining]

        if not rows:
            break

        emitted += len(rows)
        next_index += len(rows)

        yield rows

        if limit is not None and emitted >= limit:
            break

def start_prefetch_thread(
    *,
    stream,
    start_index: int,
    query_batch_size: int,
    limit: int | None,
    query_qps: float,
    max_prefetch_batches: int,
):
    queue: Queue = Queue(maxsize=max_prefetch_batches)
    sentinel = object()

    def producer():
        try:
            for batch in iter_hf_batches(
                stream=stream,
                start_index=start_index,
                query_batch_size=query_batch_size,
                limit=limit,
                query_qps=query_qps,
            ):
                queue.put(batch)
        except Exception as exception:
            queue.put(exception)
        finally:
            queue.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    return queue, sentinel, thread


def process_one_row(
    *,
    row: dict[str, Any],
    index: int,
    config,
    run_name: str,
    gcs_timeout_seconds: int,
    skip_if_exists: bool,
    rate_limiter: SimpleRateLimiter,
) -> dict[str, Any]:
    detected = detect_required_columns(row)

    audio_col = detected["audio"]
    transcript_col = detected["transcript"]
    lang_col = detected["lang"]
    gender_col = detected["gender"]

    if audio_col is None:
        raise ValueError(f"Audio column not found. Available columns: {list(row.keys())}")

    if transcript_col is None:
        raise ValueError(
            f"Transcript column not found. Available columns: {list(row.keys())}"
        )

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
    audio_blob_name = f"{run_name}/audio/{shard}/{record_id}.wav"

    with tempfile.TemporaryDirectory(prefix=f"asr_upload_{index}_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_wav_path = temp_dir_path / f"{record_id}.wav"

        source_wav_path = write_audio_value_to_wav(
            audio_value=row[audio_col],
            output_wav_path=temp_wav_path,
        )

        audio_probe = probe_audio(source_wav_path)

        gcs = get_thread_gcs_client(
            bucket_name=config.gcp_bucket_name,
            timeout_seconds=gcs_timeout_seconds,
        )

        audio_uri, uploaded_size = gcs.upload_file(
            local_path=source_wav_path,
            destination_blob_name=audio_blob_name,
            skip_if_exists=skip_if_exists,
            rate_limiter=rate_limiter,
        )

    return {
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
        "audio_format": "wav",
        "sample_rate": int(audio_probe["sample_rate"]),
        "channels": int(audio_probe["channels"]),
        "codec_name": audio_probe["codec_name"],
        "duration": audio_probe.get("duration"),
        "gcs_size_bytes": uploaded_size,
    }


def process_batch(
    *,
    batch: list[tuple[int, dict[str, Any]]],
    config,
    run_name: str,
    workers: int,
    gcs_timeout_seconds: int,
    skip_if_exists: bool,
    rate_limiter: SimpleRateLimiter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = []
    failure_rows = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                process_one_row,
                row=row,
                index=index,
                config=config,
                run_name=run_name,
                gcs_timeout_seconds=gcs_timeout_seconds,
                skip_if_exists=skip_if_exists,
                rate_limiter=rate_limiter,
            ): index
            for index, row in batch
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]

            try:
                manifest_row = future.result()
                manifest_rows.append(manifest_row)
                print(f"RUN={run_name}, SUCCESS index={index}", flush=True)
            except Exception as exception:
                failure_rows.append(
                    {
                        "source_index": index,
                        "error": str(exception),
                        "failed_at": utc_now_iso(),
                    }
                )
                print(f"RUN={run_name}, FAILED index={index}: {exception}", flush=True)

    manifest_rows.sort(key=lambda item: item["source_index"])
    failure_rows.sort(key=lambda item: item["source_index"])

    return manifest_rows, failure_rows


def run_job(job: dict[str, Any], *, reset_checkpoint: bool, dry_run: bool) -> None:
    config = build_config_for_job(job)
    run_name = job["run_name"]

    local_run_dir = Path(config.local_output_dir) / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = local_run_dir / "checkpoint.json"
    manifest_path = local_run_dir / f"{run_name}.jsonl"
    failures_path = local_run_dir / f"{run_name}_failures.jsonl"

    if reset_checkpoint:
        checkpoint = {
            "dataset_name": config.dataset_name,
            "dataset_config": config.dataset_config,
            "dataset_split": config.dataset_split,
            "run_name": run_name,
            "next_index": 0,
            "processed_count": 0,
            "failed_count": 0,
            "updated_at": utc_now_iso(),
        }
        save_checkpoint(checkpoint_path, checkpoint)

        if manifest_path.exists():
            manifest_path.unlink()

        if failures_path.exists():
            failures_path.unlink()
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint.setdefault("dataset_name", config.dataset_name)
        checkpoint.setdefault("dataset_config", config.dataset_config)
        checkpoint.setdefault("dataset_split", config.dataset_split)
        checkpoint.setdefault("run_name", run_name)
        checkpoint.setdefault("processed_count", 0)
        checkpoint.setdefault("failed_count", 0)

    start_index = int(checkpoint.get("next_index", 0))

    checkpoint["dataset_name"] = config.dataset_name
    checkpoint["dataset_config"] = config.dataset_config
    checkpoint["dataset_split"] = config.dataset_split
    checkpoint["run_name"] = run_name
    checkpoint.setdefault("processed_count", 0)
    checkpoint.setdefault("failed_count", 0)
    checkpoint.setdefault("last_completed_batch", None)
    checkpoint["updated_at"] = utc_now_iso()
    save_checkpoint(checkpoint_path, checkpoint)

    limit = None if int(job["limit"]) == -1 else int(job["limit"])

    print("=" * 100, flush=True)
    print(f"Starting job: {run_name}", flush=True)
    print(f"Dataset: {config.dataset_name}", flush=True)
    print(f"Config/language: {config.dataset_config}", flush=True)
    print(f"Split: {config.dataset_split}", flush=True)
    print(f"Bucket: gs://{config.gcp_bucket_name}", flush=True)
    print(f"Start index: {start_index}", flush=True)
    print(f"Limit: {'no limit' if limit is None else limit}", flush=True)
    print(f"Workers: {job['workers']}", flush=True)
    print(f"Batch size: {job['batch_size']}", flush=True)
    print(f"Upload QPS: {job['upload_qps']}", flush=True)
    print(f"Local checkpoint: {checkpoint_path}", flush=True)
    print("=" * 100, flush=True)

    if dry_run:
        return

    validator = FastGcsClient(
        bucket_name=config.gcp_bucket_name,
        timeout_seconds=job["gcs_timeout_seconds"],
    )
    validator.validate_bucket()

    rate_limiter = SimpleRateLimiter(job["upload_qps"])

    dataset = load_streaming_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        token=config.hf_token,
    )

    stream = dataset.skip(start_index)

    processed_this_job = 0
    failed_this_job = 0
    batch_number = 0

    prefetch_queue, prefetch_sentinel, prefetch_thread = start_prefetch_thread(
        stream=stream,
        start_index=start_index,
        query_batch_size=job["query_batch_size"],
        limit=limit,
        query_qps=job["query_qps"],
        max_prefetch_batches=job["max_prefetch_batches"],
    )

    try:
        while True:
            item = prefetch_queue.get()
            if item is prefetch_sentinel:
                break

            if isinstance(item, Exception):
                raise item

            batch = item
            batch_number += 1
            batch_start = batch[0][0]
            batch_end = batch[-1][0]

            print(
                f"Processing batch={batch_number}, indexes={batch_start}-{batch_end}, size={len(batch)}",
                flush=True,
            )

            manifest_rows, failure_rows = process_batch(
                batch=batch,
                config=config,
                run_name=run_name,
                workers=job["workers"],
                gcs_timeout_seconds=job["gcs_timeout_seconds"],
                skip_if_exists=job["skip_if_exists"],
                rate_limiter=rate_limiter,
            )

            append_jsonl_batch(manifest_path, manifest_rows)
            append_jsonl_batch(failures_path, failure_rows)

            processed_this_job += len(manifest_rows)
            failed_this_job += len(failure_rows)

            checkpoint["next_index"] = batch_end + 1
            checkpoint["processed_count"] = int(checkpoint.get("processed_count", 0)) + len(
                manifest_rows
            )
            checkpoint["failed_count"] = int(checkpoint.get("failed_count", 0)) + len(
                failure_rows
            )
            checkpoint["last_completed_batch"] = batch_number
            checkpoint["updated_at"] = utc_now_iso()

            save_checkpoint(checkpoint_path, checkpoint)

            if batch_number % job["upload_state_to_gcs_every_batches"] == 0:
                upload_state_files(
                    gcs=validator,
                    run_name=run_name,
                    manifest_path=manifest_path,
                    failures_path=failures_path,
                    checkpoint_path=checkpoint_path,
                )
                print(f"Uploaded state files after batch={batch_number}", flush=True)

            gc.collect()

    finally:
        del stream
        del dataset
        gc.collect()

    upload_state_files(
        gcs=validator,
        run_name=run_name,
        manifest_path=manifest_path,
        failures_path=failures_path,
        checkpoint_path=checkpoint_path,
    )

    print("=" * 100, flush=True)
    print(f"Finished job: {run_name}", flush=True)
    print(f"Processed in this job run: {processed_this_job}", flush=True)
    print(f"Failed in this job run: {failed_this_job}", flush=True)
    print(f"Next index: {checkpoint['next_index']}", flush=True)
    print(f"Query batch size: {job['query_batch_size']}", flush=True)
    print(f"Query QPS: {job['query_qps']}", flush=True)
    print(f"Max prefetch batches: {job['max_prefetch_batches']}", flush=True)
    print("=" * 100, flush=True)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    yaml_config = load_yaml_config(config_path)
    jobs = resolve_jobs(yaml_config)

    if args.only_run_name:
        jobs = [job for job in jobs if job["run_name"] == args.only_run_name]

        if not jobs:
            raise ValueError(f"No job found with run_name={args.only_run_name}")

    print(f"Resolved {len(jobs)} job(s).", flush=True)

    job_parallelism = max(1, int(jobs[0].get("job_parallelism", 1)))

    print(f"Job parallelism: {job_parallelism}", flush=True)

    if job_parallelism == 1:
        for job in jobs:
            run_job(job, reset_checkpoint=args.reset_checkpoint, dry_run=args.dry_run)
    else:
        with ThreadPoolExecutor(max_workers=job_parallelism) as executor:
            future_to_job = {
                executor.submit(
                    run_job,
                    job,
                    reset_checkpoint=args.reset_checkpoint,
                    dry_run=args.dry_run,
                ): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]

                try:
                    future.result()
                except Exception as exception:
                    print(
                        f"JOB FAILED run_name={job['run_name']}: {exception}",
                        flush=True,
                    )

    print("All selected jobs finished.", flush=True)


if __name__ == "__main__":
    main()