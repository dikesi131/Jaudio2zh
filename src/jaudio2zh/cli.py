from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .logging_utils import setup_logger
from .subtitles import write_ja_plain_txt, write_lrc, write_srt, write_txt
from .transcriber import WhisperTranscriber
from .translator import SakuraTranslator

_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".amr",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".ts",
    ".wav",
    ".webm",
    ".wma",
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        torch = __import__("torch")

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jaudio2zh",
        description=(
            "Transcribe Japanese audio and translate subtitles to Chinese "
            "locally."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Input audio/video file path",
    )
    input_group.add_argument(
        "--batch-input-dir",
        help="Recursively process all audio/video files in directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. If omitted, output files are saved next to "
            "each input audio file"
        ),
    )
    parser.add_argument("--log-dir", default="logs", help="Log directory")

    parser.add_argument(
        "--whisper-model",
        default="medium",
        help=(
            "openai-whisper model name (e.g. tiny/base/small/medium/large-v3) "
            "or local .pt file path"
        ),
    )
    parser.add_argument(
        "--sakura-model",
        default="auto",
        help=(
            "Sakura model id exposed by OpenAI-compatible API. "
            "Use 'auto' to fetch from /v1/models"
        ),
    )
    parser.add_argument(
        "--sakura-api-base",
        default="http://127.0.0.1:8080",
        help="OpenAI-compatible API base URL (llama.cpp server)",
    )
    parser.add_argument(
        "--sakura-api-key",
        default="",
        help="Optional API key for OpenAI-compatible endpoint",
    )

    parser.add_argument(
        "--formats",
        default="lrc,srt,txt",
        help="Comma-separated output formats: lrc,srt,txt",
    )
    parser.add_argument("--language", default="ja", help="Audio language code")
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Whisper beam size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Parallel request count for API translation",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip Japanese-to-Chinese translation",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Deprecated for openai-whisper backend (kept for compatibility)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs",
    )

    return parser.parse_args()


def _iter_audio_files(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    )


def _resolve_output_dir(input_path: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return input_path.parent


def _transcription_marker_path(
    input_path: Path, output_dir_arg: str | None
) -> Path:
    output_dir = _resolve_output_dir(input_path, output_dir_arg)
    return output_dir / f"{input_path.stem}.segments.jsonl"


def _is_already_transcribed(
    input_path: Path, output_dir_arg: str | None
) -> bool:
    return _transcription_marker_path(input_path, output_dir_arg).is_file()


def _process_one_file(
    *,
    input_path: Path,
    output_dir_arg: str | None,
    language: str,
    formats_arg: str,
    transcriber: WhisperTranscriber,
    translator: SakuraTranslator | None,
    logger,
) -> None:
    output_dir = _resolve_output_dir(input_path, output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = transcriber.transcribe(str(input_path), language=language)

    if translator is not None:
        zh_texts = translator.translate_texts(
            [seg.ja_text for seg in segments]
        )
        for seg, zh in zip(segments, zh_texts):
            seg.zh_text = zh

    stem = input_path.stem
    formats = {
        fmt.strip().lower()
        for fmt in formats_arg.split(",")
        if fmt.strip()
    }

    if "srt" in formats:
        write_srt(segments, output_dir / f"{stem}.srt")
    if "lrc" in formats:
        write_lrc(segments, output_dir / f"{stem}.lrc")
    if "txt" in formats:
        write_txt(segments, output_dir / f"{stem}.txt")

    write_ja_plain_txt(segments, output_dir / f"{stem}.ja.txt")

    jsonl_path = output_dir / f"{stem}.segments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Outputs written to: %s", output_dir.resolve())


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_file():
            print(f"Input file does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)
        input_files = [input_path]
    else:
        batch_dir = Path(args.batch_input_dir)
        if not batch_dir.exists() or not batch_dir.is_dir():
            print(
                f"Batch input directory does not exist: {batch_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        input_files = _iter_audio_files(batch_dir)
        if not input_files:
            print(
                f"No audio/video files found under: {batch_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    logger, log_file = setup_logger(Path(args.log_dir), verbose=args.verbose)

    device = _resolve_device(args.device)
    logger.info("Using device=%s", device)
    logger.info("Total files to process: %d", len(input_files))

    t0 = time.perf_counter()

    transcriber = WhisperTranscriber(
        model_name_or_path=args.whisper_model,
        device=device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        logger=logger,
    )
    translator: SakuraTranslator | None = None
    if args.no_translate:
        logger.info("Skipping translation by --no-translate")
    else:
        translator = SakuraTranslator(
            model_name_or_path=args.sakura_model,
            api_base=args.sakura_api_base,
            api_key=args.sakura_api_key,
            request_timeout=args.request_timeout,
            parallel_requests=args.batch_size,
            logger=logger,
        )

    failed_files: list[Path] = []
    skipped_files: list[Path] = []
    for idx, input_file in enumerate(input_files, start=1):
        marker_path = _transcription_marker_path(
            input_file,
            args.output_dir,
        )
        if _is_already_transcribed(input_file, args.output_dir):
            skipped_files.append(input_file)
            logger.info(
                "[%d/%d] Skip already transcribed: %s (marker: %s)",
                idx,
                len(input_files),
                input_file,
                marker_path,
            )
            continue

        logger.info(
            "[%d/%d] Processing: %s",
            idx,
            len(input_files),
            input_file,
        )
        try:
            _process_one_file(
                input_path=input_file,
                output_dir_arg=args.output_dir,
                language=args.language,
                formats_arg=args.formats,
                transcriber=transcriber,
                translator=translator,
                logger=logger,
            )
        except Exception:
            failed_files.append(input_file)
            logger.exception("Failed processing: %s", input_file)

    elapsed = time.perf_counter() - t0
    success_count = (
        len(input_files) - len(failed_files) - len(skipped_files)
    )
    logger.info(
        "Finished in %.2f seconds. Success=%d Skipped=%d Failed=%d",
        elapsed,
        success_count,
        len(skipped_files),
        len(failed_files),
    )
    if failed_files:
        for failed in failed_files:
            logger.error("Failed file: %s", failed)
    logger.info("Log file: %s", log_file.resolve())

    if failed_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
