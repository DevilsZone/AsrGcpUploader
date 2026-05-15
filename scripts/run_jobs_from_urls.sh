#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "ASR uploader container started"
echo "============================================================"

if [ -z "${URLS_JSON:-}" ]; then
  echo "ERROR: URLS_JSON is required."
  exit 1
fi

if [ -z "${SPLITS:-}" ]; then
  echo "ERROR: SPLITS is required."
  exit 1
fi

if [ -z "${GCP_BUCKET_NAME:-}" ]; then
  echo "ERROR: GCP_BUCKET_NAME is required."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is required."
  exit 1
fi

BATCH_LIMIT="${BATCH_LIMIT:-10}"
RESET_CHECKPOINT="${RESET_CHECKPOINT:-false}"
FORCE_EXIT="${FORCE_EXIT:-false}"

echo "Bucket: ${GCP_BUCKET_NAME}"
echo "Splits: ${SPLITS}"
echo "Batch limit: ${BATCH_LIMIT}"
echo "Reset checkpoint: ${RESET_CHECKPOINT}"
echo "Force exit: ${FORCE_EXIT}"
echo "URLs JSON: ${URLS_JSON}"

mkdir -p output logs

python -u scripts/build_jobs_from_urls.py \
  --urls-json "${URLS_JSON}" \
  --splits "${SPLITS}" \
  --output-file output/jobs.jsonl

echo "Generated jobs:"
cat output/jobs.jsonl

echo "============================================================"
echo "Starting jobs sequentially"
echo "============================================================"

while IFS= read -r job_json; do
  DATASET_NAME=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_name"])' "$job_json")
  DATASET_CONFIG=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_config"])' "$job_json")
  DATASET_SPLIT=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_split"])' "$job_json")
  RUN_NAME=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_name"])' "$job_json")

  echo "------------------------------------------------------------"
  echo "Running job"
  echo "Dataset: ${DATASET_NAME}"
  echo "Config: ${DATASET_CONFIG}"
  echo "Split: ${DATASET_SPLIT}"
  echo "Run name: ${RUN_NAME}"
  echo "------------------------------------------------------------"

  RESET_FLAG=""
  if [ "${RESET_CHECKPOINT}" = "true" ]; then
    RESET_FLAG="--reset-checkpoint"
  fi

  FORCE_EXIT_FLAG=""
  if [ "${FORCE_EXIT}" = "true" ]; then
    FORCE_EXIT_FLAG="--force-exit"
  fi

  python -u scripts/05_resumable_upload.py \
    --bucket "${GCP_BUCKET_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config "${DATASET_CONFIG}" \
    --dataset-split "${DATASET_SPLIT}" \
    --run-name "${RUN_NAME}" \
    --limit "${BATCH_LIMIT}" \
    ${RESET_FLAG} \
    ${FORCE_EXIT_FLAG}

  echo "Completed job: ${RUN_NAME}"

done < output/jobs.jsonl

echo "============================================================"
echo "All jobs completed"
echo "============================================================"