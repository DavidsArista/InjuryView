from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.video.ffmpeg import require_ffmpeg, run_command


class VideoMetadata(BaseModel):
    """Technical description of the supplied clip.

    `duration_s` and frame times are clip-relative. `video_path` is stored for
    the run record and must not be sent to the model.
    """

    model_config = ConfigDict(extra="forbid")

    video_path: str
    duration_s: float
    container_start_s: float = 0.0
    width: int
    height: int
    frame_rate: float
    frame_rate_raw: str
    time_base: str
    codec_name: str
    nb_frames: int | None = None
    source_offset_s: float | None = None
    probed_frame_count: int = Field(
        description="Number of video frames whose presentation timestamps were read."
    )


@dataclass(frozen=True)
class ProbedFrame:
    index: int
    pts_s: float
    time_s: float
    width: int
    height: int


def inspect_video(
    path: Path,
    *,
    source_offset_s: float | None = None,
    max_duration_s: float | None = None,
) -> tuple[VideoMetadata, list[ProbedFrame]]:
    require_ffmpeg()
    if not path.is_file():
        raise InjuryAnalysisError(f"Video file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise InjuryAnalysisError(f"Video file is empty: {path}")

    stream_payload = _ffprobe_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames,codec_name,duration",
            "-show_entries",
            "format=duration,start_time",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = stream_payload.get("streams") or []
    if not streams:
        raise InjuryAnalysisError(f"No video stream found in {path.name}")
    stream = streams[0]
    fmt = stream_payload.get("format") or {}

    width = _require_int(stream.get("width"), "width")
    height = _require_int(stream.get("height"), "height")
    rate_raw = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
    frame_rate = _parse_rate(rate_raw)
    time_base = str(stream.get("time_base") or "")
    if not time_base:
        raise InjuryAnalysisError("ffprobe did not report a video time base")
    codec_name = str(stream.get("codec_name") or "unknown")
    container_start = _optional_float(fmt.get("start_time")) or 0.0
    duration = _optional_float(fmt.get("duration"))
    if duration is None:
        duration = _optional_float(stream.get("duration"))
    nb_frames = _optional_int(stream.get("nb_frames"))
    if duration is not None and duration <= 0:
        raise InjuryAnalysisError(f"Video duration is not positive: {path.name}")
    if (
        duration is not None
        and max_duration_s is not None
        and duration > max_duration_s
    ):
        raise InjuryAnalysisError(
            f"Clip duration {duration:.1f}s exceeds the limit of {max_duration_s:.1f}s. "
            "This prototype analyzes short clips."
        )

    frames = _probe_frames(
        path,
        container_start_s=container_start,
        fallback_width=width,
        fallback_height=height,
    )
    if not frames:
        raise InjuryAnalysisError(f"No decodable video frames found in {path.name}")
    if duration is None or duration <= 0:
        duration = frames[-1].time_s
    if duration <= 0:
        raise InjuryAnalysisError(f"Video duration is not positive: {path.name}")

    metadata = VideoMetadata(
        video_path=str(path.resolve()),
        duration_s=duration,
        container_start_s=container_start,
        width=width,
        height=height,
        frame_rate=frame_rate,
        frame_rate_raw=rate_raw,
        time_base=time_base,
        codec_name=codec_name,
        nb_frames=nb_frames,
        source_offset_s=source_offset_s,
        probed_frame_count=len(frames),
    )
    return metadata, frames


def _probe_frames(
    path: Path,
    *,
    container_start_s: float,
    fallback_width: int,
    fallback_height: int,
) -> list[ProbedFrame]:
    payload = _ffprobe_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,width,height",
            "-of",
            "json",
            str(path),
        ]
    )
    raw_frames = payload.get("frames") or []
    probed: list[ProbedFrame] = []
    for raw in raw_frames:
        pts = _optional_float(raw.get("best_effort_timestamp_time"))
        if pts is None:
            pts = _optional_float(raw.get("pts_time"))
        if pts is None:
            raise InjuryAnalysisError(
                "A video frame has no presentation timestamp. "
                "This prototype only analyzes clips whose frames expose PTS values."
            )
        width = _optional_int(raw.get("width")) or fallback_width
        height = _optional_int(raw.get("height")) or fallback_height
        probed.append(
            ProbedFrame(
                index=len(probed),
                pts_s=pts,
                time_s=pts - container_start_s,
                width=width,
                height=height,
            )
        )
    probed.sort(key=lambda frame: (frame.time_s, frame.index))
    # Re-number in presentation order. Extraction uses PTS, not this index.
    return [
        ProbedFrame(
            index=index,
            pts_s=frame.pts_s,
            time_s=frame.time_s,
            width=frame.width,
            height=frame.height,
        )
        for index, frame in enumerate(probed)
    ]


def _ffprobe_json(args: list[str]) -> dict:
    stdout = run_command(args)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise InjuryAnalysisError("ffprobe returned data that was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise InjuryAnalysisError("ffprobe JSON had an unexpected shape")
    return payload


def _parse_rate(raw: str) -> float:
    if not raw or raw == "0/0":
        raise InjuryAnalysisError(f"ffprobe reported an unusable frame rate: {raw!r}")
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        den = float(denominator)
        if den == 0:
            raise InjuryAnalysisError(f"ffprobe reported an unusable frame rate: {raw!r}")
        return float(numerator) / den
    return float(raw)


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _require_int(value: object, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed <= 0:
        raise InjuryAnalysisError(f"ffprobe did not report a valid {label}")
    return parsed
