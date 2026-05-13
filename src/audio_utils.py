import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _convert_audio_array_for_soundfile(audio_array: Any) -> np.ndarray:
    """
    Converts possible torch/numpy audio arrays into a shape accepted by soundfile.

    soundfile expects:
      mono:   (samples,)
      stereo: (samples, channels)

    Some decoders return:
      (channels, samples)

    So we normalize that.
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
        # Example: (1, 123456), (2, 123456)
        if first_dim <= 8 and second_dim > first_dim:
            audio_array = audio_array.T

        return audio_array.astype(np.float32)

    raise ValueError(f"Unsupported audio array shape: {audio_array.shape}")


def write_audio_value_to_wav(audio_value: Any, output_wav_path: Path) -> Path:
    """
    Handles multiple possible audio formats:

    1. Hugging Face audio dict:
       {
         "path": "...",
         "array": numpy array,
         "sampling_rate": 48000
       }

    2. Plain local string path.

    3. AudioDecoder-like object from streaming datasets.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_value, str):
        path = Path(audio_value)

        if path.exists():
            return path

        raise ValueError(f"Audio path string does not exist locally: {audio_value}")

    if isinstance(audio_value, dict):
        audio_path = audio_value.get("path")

        if audio_path:
            path = Path(audio_path)
            if path.exists():
                return path

        if "array" in audio_value and "sampling_rate" in audio_value:
            audio_array = _convert_audio_array_for_soundfile(audio_value["array"])
            sample_rate = int(audio_value["sampling_rate"])

            sf.write(
                file=str(output_wav_path),
                data=audio_array,
                samplerate=sample_rate,
                format="WAV",
            )

            return output_wav_path

    # AudioDecoder-like object, e.g. from HF streaming dataset
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
            )

            return output_wav_path

    raise ValueError(
        f"Unsupported audio value type: {type(audio_value).__name__}. "
        f"Value preview: {str(audio_value)[:500]}"
    )


def convert_to_mp3(
    input_audio_path: Path,
    output_mp3_path: Path,
    target_sample_rate: int,
    target_channels: int,
    mp3_bitrate: str,
) -> None:
    output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio_path),
        "-ac",
        str(target_channels),
        "-ar",
        str(target_sample_rate),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        mp3_bitrate,
        str(output_mp3_path),
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
            f"FFmpeg conversion failed.\n"
            f"Input: {input_audio_path}\n"
            f"Output: {output_mp3_path}\n"
            f"Error:\n{result.stderr}"
        )


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