from __future__ import annotations

import logging
from pathlib import Path

import whisper

from .types import Segment

_KNOWN_MODEL_NAMES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "turbo",
]


class WhisperTranscriber:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        beam_size: int,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.beam_size = max(1, int(beam_size))

        model_ref = _normalize_model_reference(model_name_or_path)
        self.fp16 = device == "cuda"

        if compute_type != "int8":
            self.logger.info(
                "--compute-type is ignored by openai-whisper backend "
                "(received: %s)",
                compute_type,
            )

        self.logger.info(
            "Loading openai-whisper model=%s device=%s",
            model_ref,
            device,
        )
        self.model = whisper.load_model(model_ref, device=device)

    def transcribe(
        self, input_path: str, language: str = "ja"
    ) -> list[Segment]:
        self.logger.info("Start transcription: %s", input_path)

        result = self.model.transcribe(
            input_path,
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            fp16=self.fp16,
            verbose=False,
        )

        raw_segments = result.get("segments", []) or []
        segments: list[Segment] = []
        for seg in raw_segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()
            if end <= start or not text:
                continue
            segments.append(Segment(start=start, end=end, ja_text=text))

        self.logger.info("Transcription completed with %d segments", len(segments))
        return segments


def _normalize_model_reference(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if not value:
        return "medium"

    if value in _KNOWN_MODEL_NAMES:
        return value

    path = Path(value)
    if path.is_file() and path.suffix == ".pt":
        return str(path)

    if path.exists():
        stem = path.name.lower()
        for candidate in [
            "large-v3",
            "large-v2",
            "turbo",
            "medium",
            "small",
            "base",
            "tiny",
            "large",
        ]:
            if candidate in stem:
                return candidate

    return value
