"""Subtitle parsing and text normalization.

Everything is funnelled through SRT. Embedded tracks and sidecar files in other
formats are converted by ffmpeg first, so only one parser is needed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 00:01:02,345 --> 00:01:04,890   (WebVTT uses a dot for the fraction)
_TIME_RE = re.compile(
    r"(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})"
    r"\s*-->\s*"
    r"(?P<h2>\d{1,3}):(?P<m2>\d{2}):(?P<s2>\d{2})[,.](?P<ms2>\d{1,3})"
)

_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")          # {\an8}, {\i1}, drawing commands
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")       # <i>, <b>, <font color="...">
_ASS_BREAK_RE = re.compile(r"\\[Nnh]")                # ASS hard and soft line breaks
_WHITESPACE_RE = re.compile(r"\s+")
_MUSIC_ONLY_RE = re.compile(r"^[\s♪♫♬♩~#*_\-]*$")


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def clean_text(raw: str) -> str:
    """Strip markup and collapse whitespace.

    Override blocks are removed before HTML tags because ASS drawing commands can
    contain angle brackets that would otherwise confuse the tag pattern.
    """
    text = _ASS_OVERRIDE_RE.sub("", raw)
    text = _ASS_BREAK_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("​", "").replace("﻿", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse(content: str) -> list[Cue]:
    """Parse SRT or WebVTT text into cues.

    Blocks are split on blank lines rather than parsed by index number, because
    subtitle files in the wild frequently have missing, duplicated, or
    out-of-order indices.
    """
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []

    for block in re.split(r"\n{2,}", content):
        block = block.strip("\n")
        if not block:
            continue
        match = _TIME_RE.search(block)
        if not match:
            continue
        start = _to_seconds(match["h"], match["m"], match["s"], match["ms"])
        end = _to_seconds(match["h2"], match["m2"], match["s2"], match["ms2"])

        # Text is everything after the line holding the timestamp.
        lines = block.split("\n")
        time_line = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), 0)
        text = clean_text(" ".join(lines[time_line + 1 :]))
        if not text or _MUSIC_ONLY_RE.match(text):
            continue
        if end <= start:
            # Zero or negative length happens in malformed files. Give the cue a
            # nominal length so downstream duration maths stays sane.
            end = start + 1.0
        cues.append(Cue(start, end, text))

    cues.sort(key=lambda c: (c.start, c.end))
    return dedupe(cues)


def dedupe(cues: list[Cue]) -> list[Cue]:
    """Merge consecutive cues carrying identical text.

    Re-displaying the same line across several cues is common, especially in
    tracks generated from broadcast captions, and it would otherwise inflate the
    word count handed to the story stage.
    """
    out: list[Cue] = []
    for cue in cues:
        if out and out[-1].text == cue.text and cue.start - out[-1].end < 1.0:
            out[-1] = Cue(out[-1].start, max(out[-1].end, cue.end), cue.text)
        else:
            out.append(cue)
    return out


def stats(cues: list[Cue], runtime_s: float) -> dict[str, float | int]:
    """Summary used as a sanity signal on the chosen track.

    ``coverage_ratio`` is the fraction of runtime carrying dialogue. A very low
    value means a forced or commentary track was almost certainly picked instead
    of the full dialogue track.
    """
    spoken = sum(c.duration for c in cues)
    words = sum(len(c.text.split()) for c in cues)
    return {
        "cue_count": len(cues),
        "word_count": words,
        "spoken_seconds": round(spoken, 2),
        "coverage_ratio": round(spoken / runtime_s, 4) if runtime_s > 0 else 0.0,
        "first_cue_s": round(cues[0].start, 2) if cues else 0.0,
        "last_cue_s": round(cues[-1].end, 2) if cues else 0.0,
    }
