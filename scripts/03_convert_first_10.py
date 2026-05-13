import tempfile
from itertools import islice
from pathlib import Path

from src.audio_utils import convert_to_mp3, probe_audio, write_audio_value_to_wav
from src.config import load_config
from src.hash_utils import record_hash, transcript_hash
from src.hf_dataset import (
    detect_required_columns,
    load_streaming_dataset,
)
from src.manifest_utils import append_jsonl


def main() -> None:
    config = load_config()

    output_dir = Path(config.local_output_dir) / "local_preview"
    audio_output_dir = output_dir / "audio"
    manifest_path = output_dir / "preview_manifest.jsonl"

    if manifest_path.exists():
        manifest_path.unlink()

    dataset = load_streaming_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        token=config.hf_token,
    )

    for index, row in enumerate(islice(dataset, 10)):
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

        output_mp3_path = audio_output_dir / f"{record_id}.mp3"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_wav_path = temp_dir_path / f"{record_id}.wav"

            source_audio_path = write_audio_value_to_wav(
                audio_value=row[audio_col],
                output_wav_path=temp_wav_path,
            )

            source_probe = probe_audio(source_audio_path)

            convert_to_mp3(
                input_audio_path=source_audio_path,
                output_mp3_path=output_mp3_path,
                target_sample_rate=config.target_sample_rate,
                target_channels=config.target_channels,
                mp3_bitrate=config.mp3_bitrate,
            )

            converted_probe = probe_audio(output_mp3_path)

        manifest_row = {
            "record_id": record_id,
            "source_dataset": config.dataset_name,
            "source_config": config.dataset_config,
            "source_split": config.dataset_split,
            "source_index": index,
            "local_audio_path": str(output_mp3_path),
            "transcript": transcript,
            "lang": lang,
            "gender": gender,
            "transcript_hash": transcript_hash(transcript),
            "source_audio_probe": source_probe,
            "converted_audio_probe": converted_probe,
        }

        append_jsonl(manifest_path, manifest_row)

        print(f"Converted {index + 1}/10")
        print(f"  MP3: {output_mp3_path}")
        print(f"  Source audio: {source_probe}")
        print(f"  Converted audio: {converted_probe}")

    print("=" * 80)
    print(f"Done. Preview files saved under: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()