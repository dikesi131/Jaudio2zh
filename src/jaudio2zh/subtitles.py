from __future__ import annotations

from pathlib import Path

from .types import Segment


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_lrc_time(seconds: float) -> str:
    centis = int(round(seconds * 100))
    m = centis // 6000
    centis %= 6000
    s = centis // 100
    cs = centis % 100
    return f"{m:02d}:{s:02d}.{cs:02d}"


def write_srt(segments: list[Segment], output_path: Path) -> None:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.zh_text or seg.ja_text
        lines.extend(
            [
                str(idx),
                f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}",
                text.strip(),
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_lrc(segments: list[Segment], output_path: Path) -> None:
    lines = []
    for seg in segments:
        text = (seg.zh_text or seg.ja_text).strip()
        lines.append(f"[{_fmt_lrc_time(seg.start)}]{text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_txt(segments: list[Segment], output_path: Path) -> None:
    lines = [f"{seg.ja_text}\t{seg.zh_text}" for seg in segments]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ja_plain_txt(segments: list[Segment], output_path: Path) -> None:
    lines = [seg.ja_text.strip() for seg in segments if seg.ja_text.strip()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
