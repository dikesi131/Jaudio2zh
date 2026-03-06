from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    ja_text: str
    zh_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
