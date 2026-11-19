from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.video.ffmpeg import run_command
from injury_analysis.video.metadata import ProbedFrame, VideoMetadata

Sampling = Literal["broad", "focused", "both"]


class FrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    source_frame_index: int
    pts_s: float
    time_s: float
    source_time_s: float | None = None
    path: str
    width: int
    height: int
    sampling: Sampling
    clip_id: str = ""


class ContactSheetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_id: str
    path: str
    frame_ids: list[str]


class ClipRecord(BaseModel):
    """One camera angle. `video_path` is for the human record and must not be sent to the model."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    video_path: str = ""
    duration_s: float
    frame_rate: float
    width: int
    height: int
    focus_start_s: float | None = None
    focus_end_s: float | None = None
    video_sha256: str | None = None


class FrameManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broad_fps: float
    focused_fps: float
    focus_start_s: float | None = None
    focus_end_s: float | None = None
    source_offset_s: float | None = None
    video_sha256: str | None = None
    frames: list[FrameRecord]
    contact_sheets: list[ContactSheetRecord] = Field(default_factory=list)
    sheet_columns: int | None = None
    frames_per_sheet: int | None = None
    clips: list[ClipRecord] = Field(default_factory=list)
    total_frame_count: int | None = Field(
        default=None,
        description="Set only on a stage's narrowed copy: how many frames were extracted in all.",
    )

    def by_id(self) -> dict[str, FrameRecord]:
        return {frame.frame_id: frame for frame in self.frames}

    def frame(self, frame_id: str) -> FrameRecord:
        try:
            return self.by_id()[frame_id]
        except KeyError as exc:
            raise InjuryAnalysisError(f"Unknown frame ID: {frame_id}") from exc


class SelectedFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_frame_index: int
    pts_s: float
    time_s: float
    width: int
    height: int
    sampling: Sampling


def target_times(start: float, end: float, fps: float) -> list[float]:
    """Clip-relative sample targets at a fixed rate, inclusive of the window start."""
    if fps <= 0:
        raise InjuryAnalysisError("Sampling rate must be positive")
    if end < start:
        raise InjuryAnalysisError("Sampling window ends before it starts")
    interval = 1.0 / fps
    times: list[float] = []
    cursor = start
    for _ in range(100_000):
        if cursor > end + 1e-9:
            break
        times.append(round(cursor, 6))
        nxt = round(cursor + interval, 6)
        if nxt <= cursor:
            raise InjuryAnalysisError("Sampling interval did not advance")
        cursor = nxt
    else:
        raise InjuryAnalysisError("Sampling rate produced too many targets")
    if not times:
        times = [round(start, 6)]
    if end - times[-1] > interval * 0.5:
        times.append(round(end, 6))
    deduped: list[float] = []
    for value in times:
        if not deduped or abs(deduped[-1] - value) > 1e-6:
            deduped.append(value)
    return deduped


def select_frames(
    probed: list[ProbedFrame],
    *,
    broad_fps: float,
    focused_fps: float,
    focus_start_s: float | None,
    focus_end_s: float | None,
    max_frames: int,
) -> list[SelectedFrame]:
    if not probed:
        raise InjuryAnalysisError("Cannot sample a video with no frames")
    if max_frames < 1:
        raise InjuryAnalysisError("max_frames must be at least 1")
    if (focus_start_s is None) ^ (focus_end_s is None):
        raise InjuryAnalysisError("Provide both focus start and focus end, or neither")

    last_time = probed[-1].time_s
    first_time = probed[0].time_s
    broad_targets = target_times(first_time, last_time, broad_fps)
    broad = _assign(probed, broad_targets, "broad sampling")
    focused: dict[int, ProbedFrame] = {}
    if focus_start_s is not None and focus_end_s is not None:
        window_end = min(focus_end_s, last_time)
        window_start = focus_start_s
        if window_end < window_start:
            raise InjuryAnalysisError(
                "The focused window ends before the last available frame in that window"
            )
        focused_targets = target_times(window_start, window_end, focused_fps)
        focused = _assign(probed, focused_targets, "focused sampling")

    by_index = {frame.index: frame for frame in probed}
    selected: list[SelectedFrame] = []
    for index in sorted(set(broad) | set(focused), key=lambda i: (by_index[i].time_s, i)):
        labels = set()
        if index in broad:
            labels.add("broad")
        if index in focused:
            labels.add("focused")
        sampling: Sampling
        if labels == {"broad", "focused"}:
            sampling = "both"
        elif "focused" in labels:
            sampling = "focused"
        else:
            sampling = "broad"
        frame = by_index[index]
        selected.append(
            SelectedFrame(
                source_frame_index=frame.index,
                pts_s=frame.pts_s,
                time_s=frame.time_s,
                width=frame.width,
                height=frame.height,
                sampling=sampling,
            )
        )
    if len(selected) > max_frames:
        raise InjuryAnalysisError(
            f"Sampling selected {len(selected)} frames, which exceeds max_frames={max_frames}. "
            "Lower the sampling rates, shorten the focused window, or raise --max-frames."
        )
    return selected


def extract_frames(
    video_path: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    probed: list[ProbedFrame],
    *,
    broad_fps: float,
    focused_fps: float,
    focus_start_s: float | None,
    focus_end_s: float | None,
    max_frames: int,
    source_offset_s: float | None,
    clip_id: str = "",
    start_number: int = 1,
    reset_dir: bool = True,
) -> FrameManifest:
    selected = select_frames(
        probed,
        broad_fps=broad_fps,
        focused_fps=focused_fps,
        focus_start_s=focus_start_s,
        focus_end_s=focus_end_s,
        max_frames=max_frames,
    )
    frames_dir = output_dir / "frames"
    if reset_dir and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _extract_selected_jpegs(
        video_path,
        frames_dir,
        selected,
        epsilon=_source_epsilon(probed),
        start_number=start_number,
    )

    records: list[FrameRecord] = []
    for number, frame in enumerate(selected, start=start_number):
        frame_id = f"f{number:03d}"
        filename = f"{frame_id}.jpg"
        source_time = None if source_offset_s is None else frame.time_s + source_offset_s
        records.append(
            FrameRecord(
                frame_id=frame_id,
                source_frame_index=frame.source_frame_index,
                pts_s=frame.pts_s,
                time_s=frame.time_s,
                source_time_s=source_time,
                path=f"frames/{filename}",
                width=frame.width,
                height=frame.height,
                sampling=frame.sampling,
                clip_id=clip_id,
            )
        )
    return FrameManifest(
        broad_fps=broad_fps,
        focused_fps=focused_fps,
        focus_start_s=focus_start_s,
        focus_end_s=focus_end_s,
        source_offset_s=source_offset_s if source_offset_s is not None else metadata.source_offset_s,
        frames=records,
    )


def _assign(probed: list[ProbedFrame], targets: list[float], label: str) -> dict[int, ProbedFrame]:
    if len(targets) > len(probed):
        raise InjuryAnalysisError(
            f"{label.capitalize()} asked for {len(targets)} samples, "
            f"but the clip only has {len(probed)} frames. Lower the sampling rate."
        )
    used: set[int] = set()
    chosen: dict[int, ProbedFrame] = {}
    for target in targets:
        available = [frame for frame in probed if frame.index not in used]
        best = min(available, key=lambda frame: (abs(frame.time_s - target), frame.index))
        used.add(best.index)
        chosen[best.index] = best
    return chosen


def _extract_selected_jpegs(
    video_path: Path,
    frames_dir: Path,
    selected: list[SelectedFrame],
    *,
    epsilon: float,
    start_number: int = 1,
) -> None:
    clauses = []
    for frame in selected:
        low = frame.pts_s - epsilon
        high = frame.pts_s + epsilon
        clauses.append(f"between(t\\,{low:.6f}\\,{high:.6f})")
    select_expr = "+".join(clauses)
    pattern = frames_dir / "tmp_%04d.jpg"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            # Keep original timestamps so the select filter's t matches probed PTS
            # when the container does not start at zero.
            "-copyts",
            "-i",
            str(video_path),
            "-filter:v",
            f"select='{select_expr}'",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(pattern),
        ]
    )
    extracted = sorted(frames_dir.glob("tmp_*.jpg"))
    if len(extracted) != len(selected):
        raise InjuryAnalysisError(
            f"Frame extraction produced {len(extracted)} images for {len(selected)} "
            "selected presentation timestamps. No frames were renamed."
        )
    for image, frame_number in zip(
        extracted,
        range(start_number, start_number + len(selected)),
        strict=True,
    ):
        image.rename(frames_dir / f"f{frame_number:03d}.jpg")


def _source_epsilon(probed: list[ProbedFrame]) -> float:
    """Half-open window small enough to contain one source frame and not its neighbor."""
    times = [frame.pts_s for frame in probed]
    gaps = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not gaps:
        return 0.001
    return min(gaps) * 0.45
