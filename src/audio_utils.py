import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _convert_audio_array_for_soundfile(audio_array: Any) -> np.ndarray:
    """
    soundfile expects:
      mono:   (samples,)
      stereo: (samples, channels)

    Some decoders return:
      (channels, samples)

    This function normalizes shape safely.
    """
    if hasattr(audio_array, "detach"):
        audio_array = audio_array.detach().cpu().numpy()

    audio_array = np.asarray(audio_array)

    if audio_array.ndim == 0:
        raise ValueError("Audio array is scalar; cannot write audio.")

    if audio_array.ndim == 1:
        return audio_array.astype(np.float32)

    if audio_array.ndim == 2:
        first_dim = audio_array.shape[0]
        second_dim = audio_array.shape[1]

        # Likely shape: (channels, samples)
        if first_dim <= 8 and second_dim > first_dim:
            audio_array = audio_array.T

        return audio_array.astype(np.float32)

    raise ValueError(f"Unsupported audio array shape: {audio_array.shape}")


def probe_audio(audio_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration,bit_rate",
        "-of",
        "json",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}: {result.stderr}")

    data = json.loads(result.stdout)

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream found in {audio_path}")

    return streams[0]


def is_wav_file(audio_path: Path) -> bool:
    """
    Checks using ffprobe codec/container behavior.
    Common WAV PCM codecs include:
      pcm_s16le, pcm_s24le, pcm_f32le, pcm_s32le
    """
    try:
        probe = probe_audio(audio_path)
    except Exception:
        return False

    codec_name = str(probe.get("codec_name", "")).lower()
    return codec_name.startswith("pcm_")


def get_audio_stats_from_wav(audio_path: Path) -> dict:
    data, sample_rate = sf.read(str(audio_path), always_2d=True)

    channel_count = data.shape[1]
    channel_stats = []

    for channel_index in range(channel_count):
        channel = data[:, channel_index]

        rms = float(np.sqrt(np.mean(np.square(channel))))
        peak = float(np.max(np.abs(channel)))

        channel_stats.append(
            {
                "channel_index": channel_index,
                "rms": rms,
                "peak": peak,
            }
        )

    return {
        "sample_rate": int(sample_rate),
        "channels": int(channel_count),
        "samples": int(data.shape[0]),
        "duration_seconds": float(data.shape[0] / sample_rate),
        "channel_stats": channel_stats,
    }


def write_audio_value_to_wav(audio_value: Any, output_wav_path: Path) -> Path:
    """
    Converts/keeps audio as WAV.

    Handles:
    1. Plain local path string.
    2. Hugging Face audio dict.
    3. AudioDecoder-like object from streaming datasets.

    Always returns a local WAV path.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Case 1: local path string
    if isinstance(audio_value, str):
        source_path = Path(audio_value)

        if not source_path.exists():
            raise ValueError(f"Audio path string does not exist locally: {audio_value}")

        if is_wav_file(source_path):
            return source_path

        # If it is not WAV, convert/copy into WAV using ffmpeg.
        convert_any_audio_to_wav(source_path, output_wav_path)
        return output_wav_path

    # Case 2: HF-style dict
    if isinstance(audio_value, dict):
        audio_path = audio_value.get("path")

        if audio_path:
            source_path = Path(audio_path)
            if source_path.exists():
                if is_wav_file(source_path):
                    return source_path

                convert_any_audio_to_wav(source_path, output_wav_path)
                return output_wav_path

        if "array" in audio_value and "sampling_rate" in audio_value:
            audio_array = _convert_audio_array_for_soundfile(audio_value["array"])
            sample_rate = int(audio_value["sampling_rate"])

            sf.write(
                file=str(output_wav_path),
                data=audio_array,
                samplerate=sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            return output_wav_path

    # Case 3: AudioDecoder-like object
    if hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()

        if hasattr(samples, "data") and hasattr(samples, "sample_rate"):
            audio_array = _convert_audio_array_for_soundfile(samples.data)
            sample_rate = int(samples.sample_rate)

            sf.write(
                file=str(output_wav_path),
                data=audio_array,
                samplerate=sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            return output_wav_path

    if hasattr(audio_value, "read"):
        decoded = audio_value.read()

        if isinstance(decoded, dict) and "array" in decoded and "sampling_rate" in decoded:
            audio_array = _convert_audio_array_for_soundfile(decoded["array"])
            sample_rate = int(decoded["sampling_rate"])

            sf.write(
                file=str(output_wav_path),
                data=audio_array,
                samplerate=sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            return output_wav_path

    raise ValueError(
        f"Unsupported audio value type: {type(audio_value).__name__}. "
        f"Value preview: {str(audio_value)[:500]}"
    )


def convert_any_audio_to_wav(input_audio_path: Path, output_wav_path: Path) -> None:
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio_path),
        "-codec:a",
        "pcm_s16le",
        str(output_wav_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg WAV conversion failed.\n"
            f"Input: {input_audio_path}\n"
            f"Output: {output_wav_path}\n"
            f"Error:\n{result.stderr}"
        )


def copy_wav_to_destination(source_wav_path: Path, destination_wav_path: Path) -> Path:
    destination_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if source_wav_path.resolve() != destination_wav_path.resolve():
        shutil.copy2(source_wav_path, destination_wav_path)

    return destination_wav_path