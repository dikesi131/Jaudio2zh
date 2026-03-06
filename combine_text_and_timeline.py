from __future__ import annotations

import argparse
import re
from pathlib import Path


def _read_non_empty_lines(path: Path, encoding: str) -> list[str]:
    lines = path.read_text(encoding=encoding).splitlines()
    return [line.strip() for line in lines if line.strip()]


def _resolve_output_path(timeline_path: Path, output_file: str | None) -> Path:
    if output_file:
        return Path(output_file)
    suffix = timeline_path.suffix
    return timeline_path.with_name(f"{timeline_path.stem}.zh{suffix}")


def _parse_srt_blocks(content: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in content.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)

    if current:
        blocks.append(current)
    return blocks


def _combine_srt(
    timeline_path: Path,
    zh_lines: list[str],
    output_path: Path,
    strict: bool,
    encoding: str,
) -> tuple[int, int]:
    blocks = _parse_srt_blocks(timeline_path.read_text(encoding=encoding))
    subtitle_blocks: list[list[str]] = []
    for block in blocks:
        timeline_line_idx = -1
        for idx, line in enumerate(block):
            if "-->" in line:
                timeline_line_idx = idx
                break
        if timeline_line_idx >= 0:
            subtitle_blocks.append(block)

    timeline_count = len(subtitle_blocks)
    zh_count = len(zh_lines)
    if strict and zh_count != timeline_count:
        raise ValueError(
            "Line count mismatch: "
            f"zh_lines={zh_count}, timeline_segments={timeline_count}"
        )

    out_blocks: list[str] = []
    zh_idx = 0
    for block in blocks:
        timeline_line_idx = -1
        for idx, line in enumerate(block):
            if "-->" in line:
                timeline_line_idx = idx
                break

        if timeline_line_idx < 0:
            out_blocks.append("\n".join(block))
            continue

        prefix = block[: timeline_line_idx + 1]
        original_parts = block[timeline_line_idx + 1:]
        original_text = " ".join(part.strip() for part in original_parts)
        if zh_idx < zh_count:
            merged_text = zh_lines[zh_idx]
        else:
            merged_text = original_text
        zh_idx += 1
        out_blocks.append("\n".join(prefix + [merged_text]))

    output_path.write_text("\n\n".join(out_blocks) + "\n", encoding=encoding)
    return timeline_count, zh_count


def _combine_lrc(
    timeline_path: Path,
    zh_lines: list[str],
    output_path: Path,
    strict: bool,
    encoding: str,
) -> tuple[int, int]:
    timestamp_pattern = re.compile(
        r"^(\[[0-9]{1,2}:[0-9]{2}(?:\.[0-9]{1,3})?\])(.*)$"
    )
    lines = timeline_path.read_text(encoding=encoding).splitlines()

    timeline_count = 0
    for line in lines:
        if timestamp_pattern.match(line):
            timeline_count += 1

    zh_count = len(zh_lines)
    if strict and zh_count != timeline_count:
        raise ValueError(
            "Line count mismatch: "
            f"zh_lines={zh_count}, timeline_segments={timeline_count}"
        )

    out_lines: list[str] = []
    zh_idx = 0
    for line in lines:
        match = timestamp_pattern.match(line)
        if not match:
            out_lines.append(line)
            continue

        timestamp, original_text = match.group(1), match.group(2).strip()
        if zh_idx < zh_count:
            merged_text = zh_lines[zh_idx]
        else:
            merged_text = original_text
        zh_idx += 1
        out_lines.append(f"{timestamp}{merged_text}")

    output_path.write_text("\n".join(out_lines) + "\n", encoding=encoding)
    return timeline_count, zh_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Chinese subtitle lines (no timeline) "
            "with a timeline file "
            "from transcription. Supports .srt and .lrc"
        )
    )
    parser.add_argument(
        "--timeline-file",
        required=True,
        help="Timeline subtitle file path (.srt or .lrc)",
    )
    parser.add_argument(
        "--zh-text-file",
        required=True,
        help="Chinese text file path (one segment per non-empty line)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output file path. Defaults to <timeline_stem>.zh.<ext>",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help=(
            "Allow zh line count to differ from timeline segment count. "
            "When zh lines are fewer, remaining timeline texts are kept."
        ),
    )
    args = parser.parse_args()

    timeline_path = Path(args.timeline_file)
    zh_text_path = Path(args.zh_text_file)
    output_path = _resolve_output_path(timeline_path, args.output_file)

    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")
    if not zh_text_path.exists():
        raise FileNotFoundError(f"Chinese text file not found: {zh_text_path}")

    zh_lines = _read_non_empty_lines(zh_text_path, args.encoding)
    strict = not args.allow_mismatch

    suffix = timeline_path.suffix.lower()
    if suffix == ".srt":
        timeline_count, zh_count = _combine_srt(
            timeline_path,
            zh_lines,
            output_path,
            strict,
            args.encoding,
        )
    elif suffix == ".lrc":
        timeline_count, zh_count = _combine_lrc(
            timeline_path,
            zh_lines,
            output_path,
            strict,
            args.encoding,
        )
    else:
        raise ValueError(
            f"Unsupported timeline file format: {timeline_path.suffix}. "
            "Use .srt or .lrc"
        )

    print(f"Timeline file: {timeline_path}")
    print(f"Chinese text file: {zh_text_path}")
    print(f"Output file: {output_path}")
    print(f"Timeline segments: {timeline_count}")
    print(f"Chinese lines: {zh_count}")
    if zh_count != timeline_count:
        print("Counts differ and were merged with --allow-mismatch behavior.")


if __name__ == "__main__":
    main()
