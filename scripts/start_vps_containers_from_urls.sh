#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/asr-gcp-uploader}"
IMAGE_NAME="${IMAGE_NAME:-asr-gcp-uploader:latest}"

URLS_JSON="${URLS_JSON:?URLS_JSON is required}"
SPLITS="${SPLITS:?SPLITS is required}"
GCP_BUCKET_NAME="${GCP_BUCKET_NAME:?GCP_BUCKET_NAME is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"

BATCH_LIMIT="${BATCH_LIMIT:-10}"
RESET_CHECKPOINT="${RESET_CHECKPOINT:-false}"
FORCE_EXIT="${FORCE_EXIT:-true}"
MAX_PARALLEL_CONTAINERS="${MAX_PARALLEL_CONTAINERS:-2}"

cd "${APP_DIR}"

mkdir -p output credentials logs

echo "============================================================"
echo "Building Docker image: ${IMAGE_NAME}"
echo "============================================================"

docker build -t "${IMAGE_NAME}" .

echo "============================================================"
echo "Generating jobs from URLs"
echo "============================================================"

python3 scripts/build_jobs_from_urls.py \
  --urls-json "${URLS_JSON}" \
  --splits "${SPLITS}" \
  --output-file output/jobs.jsonl

echo "Generated jobs:"
cat output/jobs.jsonl

echo "============================================================"
echo "Starting one container per job"
echo "Max parallel containers: ${MAX_PARALLEL_CONTAINERS}"
echo "============================================================"

sanitize_container_name() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's#[^a-z0-9_.-]#-#g' \
    | sed 's#--*#-#g' \
    | sed 's#^-##' \
    | sed 's#-$##'
}

wait_for_slot() {
  while true; do
    running_count=$(docker ps \
      --filter "label=asr-uploader=true" \
      --filter "status=running" \
      --format "{{.Names}}" \
      | wc -l \
      | tr -d ' ')

    if [ "${running_count}" -lt "${MAX_PARALLEL_CONTAINERS}" ]; then
      break
    fi

    echo "Currently running ${running_count} ASR containers. Waiting for a free slot..."
    sleep 30
  done
}

while IFS= read -r job_json; do
  DATASET_NAME=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_name"])' "$job_json")
  DATASET_CONFIG=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_config"])' "$job_json")
  DATASET_SPLIT=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_split"])' "$job_json")
  RUN_NAME=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["run_name"])' "$job_json")

  CONTAINER_NAME=$(sanitize_container_name "asr-${RUN_NAME}")

  echo "------------------------------------------------------------"
  echo "Preparing container: ${CONTAINER_NAME}"
  echo "Dataset: ${DATASET_NAME}"
  echo "Config: ${DATASET_CONFIG}"
  echo "Split: ${DATASET_SPLIT}"
  echo "Run name: ${RUN_NAME}"
  echo "------------------------------------------------------------"

  wait_for_slot

  echo "Removing old stopped container if exists: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  RESET_FLAG=""
  if [ "${RESET_CHECKPOINT}" = "true" ]; then
    RESET_FLAG="--reset-checkpoint"
  fi

  FORCE_EXIT_FLAG=""
  if [ "${FORCE_EXIT}" = "true" ]; then
    FORCE_EXIT_FLAG="--force-exit"
  fi

  echo "Starting container: ${CONTAINER_NAME}"

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --label "asr-uploader=true" \
    --label "asr-run-name=${RUN_NAME}" \
    --restart "no" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e GCP_BUCKET_NAME="${GCP_BUCKET_NAME}" \
    -e GCP_RUN_NAME="${RUN_NAME}" \
    -e DATASET_NAME="${DATASET_NAME}" \
    -e DATASET_CONFIG="${DATASET_CONFIG}" \
    -e DATASET_SPLIT="${DATASET_SPLIT}" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/gcp-key.json" \
    -e PYTHONPATH="/app" \
    -e PYTHONUNBUFFERED="1" \
    -v "${APP_DIR}/output:/app/output" \
    -v "${APP_DIR}/credentials:/app/credentials:ro" \
    -v "${APP_DIR}/logs:/app/logs" \
    "${IMAGE_NAME}" \
    python -u scripts/05_resumable_upload.py \
      --bucket "${GCP_BUCKET_NAME}" \
      --dataset-name "${DATASET_NAME}" \
      --dataset-config "${DATASET_CONFIG}" \
      --dataset-split "${DATASET_SPLIT}" \
      --run-name "${RUN_NAME}" \
      --limit "${BATCH_LIMIT}" \
      ${RESET_FLAG} \
      ${FORCE_EXIT_FLAG}

done < output/jobs.jsonl

echo "============================================================"
echo "All containers submitted."
echo "Currently running ASR containers:"
docker ps --filter "label=asr-uploader=true"
echo "============================================================"