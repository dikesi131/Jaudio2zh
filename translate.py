from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="translate.py",
        description=(
            "Translate text files sentence-by-sentence (one line per "
            "sentence) using Sakura API."
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Single text file to translate",
    )
    input_group.add_argument(
        "--batch-input-dir",
        help="Recursively translate all .txt files under this directory",
    )

    parser.add_argument(
        "--sakura-model",
        default="auto",
        help="Sakura model id exposed by OpenAI-compatible API",
    )
    parser.add_argument(
        "--sakura-api-base",
        default="http://127.0.0.1:8080",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--sakura-api-key",
        default="",
        help="Optional API key for endpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Parallel request count for API translation",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs",
    )
    return parser.parse_args()


def build_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("translate_text")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def resolve_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input:
        file_path = Path(args.input)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {file_path}")
        return [file_path]

    root_dir = Path(args.batch_input_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(
            f"Batch input directory does not exist: {root_dir}"
        )

    files = sorted(
        path
        for path in root_dir.rglob("*.txt")
        if path.is_file() and not path.name.endswith("_translated.txt")
    )
    if not files:
        raise FileNotFoundError(
            f"No source .txt files found under: {root_dir}"
        )
    return files


def output_path_for(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_translated.txt")


def is_already_translated(input_file: Path) -> bool:
    return output_path_for(input_file).is_file()


def translate_file(
    input_file: Path,
    translator,
    logger: logging.Logger,
) -> None:
    source_lines = input_file.read_text(encoding="utf-8").splitlines()

    non_empty_indices: list[int] = []
    text_batch: list[str] = []
    for idx, line in enumerate(source_lines):
        text = line.strip()
        if text:
            non_empty_indices.append(idx)
            text_batch.append(text)

    translated_lines = list(source_lines)
    if text_batch:
        translated_texts = translator.translate_texts(text_batch)
        for idx, translated in zip(non_empty_indices, translated_texts):
            translated_lines[idx] = translated

    output_path = output_path_for(input_file)
    output_path.write_text(
        "\n".join(translated_lines) + "\n",
        encoding="utf-8",
    )
    logger.info("Output written: %s", output_path)


def main() -> None:
    args = parse_args()
    logger = build_logger(verbose=args.verbose)

    try:
        input_files = resolve_input_files(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    logger.info("Total files to translate: %d", len(input_files))

    skipped_files: list[Path] = []
    pending_files: list[Path] = []
    for input_file in input_files:
        if is_already_translated(input_file):
            skipped_files.append(input_file)
            logger.info(
                "Skip already translated: %s",
                output_path_for(input_file),
            )
            continue
        pending_files.append(input_file)

    if not pending_files:
        logger.info(
            "Finished. Success=0 Skipped=%d Failed=0",
            len(skipped_files),
        )
        return

    translator_module = importlib.import_module("jaudio2zh.translator")
    sakura_translator_cls = getattr(translator_module, "SakuraTranslator")

    translator = sakura_translator_cls(
        model_name_or_path=args.sakura_model,
        api_base=args.sakura_api_base,
        api_key=args.sakura_api_key,
        request_timeout=args.request_timeout,
        parallel_requests=args.batch_size,
        logger=logger,
    )

    failed_files: list[Path] = []
    for idx, input_file in enumerate(pending_files, start=1):
        logger.info(
            "[%d/%d] Translating file: %s",
            idx,
            len(pending_files),
            input_file,
        )
        try:
            translate_file(input_file, translator, logger)
        except Exception:
            failed_files.append(input_file)
            logger.exception("Failed translating: %s", input_file)

    success_count = len(pending_files) - len(failed_files)
    logger.info(
        "Finished. Success=%d Skipped=%d Failed=%d",
        success_count,
        len(skipped_files),
        len(failed_files),
    )
    if failed_files:
        for failed in failed_files:
            logger.error("Failed file: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
