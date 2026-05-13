import tempfile
from itertools import islice
from pathlib import Path

from src.audio_utils import (
    copy_wav_to_destination,
    get_audio_stats_from_wav,
    probe_audio,
    write_audio_value_to_wav,
)
from src.config import load_config
from src.hash_utils import record_hash, transcript_hash
from src.hf_dataset import detect_required_columns, load_streaming_dataset
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

        final_wav_path = audio_output_dir / f"{record_id}.wav"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_wav_path = temp_dir_path / f"{record_id}.wav"

            source_wav_path = write_audio_value_to_wav(
                audio_value=row[audio_col],
                output_wav_path=temp_wav_path,
            )

            source_probe = probe_audio(source_wav_path)
            source_stats = get_audio_stats_from_wav(source_wav_path)

            copy_wav_to_destination(
                source_wav_path=source_wav_path,
                destination_wav_path=final_wav_path,
            )

            final_probe = probe_audio(final_wav_path)
            final_stats = get_audio_stats_from_wav(final_wav_path)

        manifest_row = {
            "record_id": record_id,
            "source_dataset": config.dataset_name,
            "source_config": config.dataset_config,
            "source_split": config.dataset_split,
            "source_index": index,
            "local_audio_path": str(final_wav_path),
            "transcript": transcript,
            "lang": lang,
            "gender": gender,
            "transcript_hash": transcript_hash(transcript),
            "audio_format": "wav",
            "source_audio_probe": source_probe,
            "source_audio_stats": source_stats,
            "final_audio_probe": final_probe,
            "final_audio_stats": final_stats,
        }

        append_jsonl(manifest_path, manifest_row)

        print(f"Saved WAV {index + 1}/10")
        print(f"  WAV: {final_wav_path}")
        print(f"  Source probe: {source_probe}")
        print(f"  Source stats: {source_stats}")
        print(f"  Final probe: {final_probe}")

    print("=" * 80)
    print(f"Done. Preview WAV files saved under: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()