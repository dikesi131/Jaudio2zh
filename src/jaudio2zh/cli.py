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
    parser.add_argument(
        "--input",
        required=True,
        help="Input audio/video file path",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory",
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


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_file = setup_logger(Path(args.log_dir), verbose=args.verbose)

    device = _resolve_device(args.device)
    logger.info("Using device=%s", device)

    t0 = time.perf_counter()

    transcriber = WhisperTranscriber(
        model_name_or_path=args.whisper_model,
        device=device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        logger=logger,
    )
    segments = transcriber.transcribe(str(input_path), language=args.language)

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
        zh_texts = translator.translate_texts(
            [seg.ja_text for seg in segments]
        )

        for seg, zh in zip(segments, zh_texts):
            seg.zh_text = zh

    stem = input_path.stem
    formats = {
        fmt.strip().lower()
        for fmt in args.formats.split(",")
        if fmt.strip()
    }

    if "srt" in formats:
        write_srt(segments, output_dir / f"{stem}.srt")
    if "lrc" in formats:
        write_lrc(segments, output_dir / f"{stem}.lrc")
    if "txt" in formats:
        write_txt(segments, output_dir / f"{stem}.txt")

    # Always export Japanese subtitle lines without timeline for reference.
    write_ja_plain_txt(segments, output_dir / f"{stem}.ja.txt")

    jsonl_path = output_dir / f"{stem}.segments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    logger.info("Finished successfully in %.2f seconds", elapsed)
    logger.info("Outputs written to: %s", output_dir.resolve())
    logger.info("Log file: %s", log_file.resolve())


if __name__ == "__main__":
    main()
