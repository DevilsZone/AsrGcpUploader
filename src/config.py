import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class PipelineConfig:
    hf_token: str | None

    gcp_bucket_name: str
    gcp_run_name: str

    dataset_name: str
    dataset_config: str
    dataset_split: str

    target_sample_rate: int
    target_channels: int
    mp3_bitrate: str

    local_output_dir: str = "output"


def load_config() -> PipelineConfig:
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    run_name = os.getenv("GCP_RUN_NAME")

    if not bucket_name:
        raise ValueError("Missing GCP_BUCKET_NAME in .env or environment.")

    if not run_name:
        raise ValueError("Missing GCP_RUN_NAME in .env or environment.")

    return PipelineConfig(
        hf_token=os.getenv("HF_TOKEN"),
        gcp_bucket_name=bucket_name,
        gcp_run_name=run_name,
        dataset_name=os.getenv("DATASET_NAME", "ai4bharat/IndicVoices"),
        dataset_config=os.getenv("DATASET_CONFIG", "hindi"),
        dataset_split=os.getenv("DATASET_SPLIT", "valid"),
        target_sample_rate=int(os.getenv("TARGET_SAMPLE_RATE", "16000")),
        target_channels=int(os.getenv("TARGET_CHANNELS", "1")),
        mp3_bitrate=os.getenv("MP3_BITRATE", "64k"),
    )