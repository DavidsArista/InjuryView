from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.model_adapter import ImagePayload
from injury_analysis.video.contact_sheet import build_contact_sheets
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def subject_block(subjects: list[str]) -> str:
    if not subjects:
        return ""
    lines = [
        "Subject to follow. This only identifies who to watch.",
        "It is not a statement about contact, a mechanism, or an injury.",
        "Player to follow:",
        subjects[0],
    ]
    if len(subjects) > 1:
        lines.append("Other people in the frame, listed only so they are not confused with the player to follow:")
        lines.extend(f"- {item}" for item in subjects[1:])
    lines.append(
        "Follow the player to follow for the whole clip. Other players may share a color, "
        "including white pants and a white jersey. Use the role, number, and full uniform "
        "description together. If a different player goes to the ground, keep this timeline "
        "on the player to follow. When an observation is about that player, copy the "
        "description above into the player field."
    )
    return "\n".join(lines) + "\n\n"


def metadata_block(metadata: VideoMetadata, manifest: FrameManifest) -> str:
    if len(manifest.clips) > 1:
        return _multi_angle_block(manifest)
    focus_start = "none" if manifest.focus_start_s is None else f"{manifest.focus_start_s:.6f}"
    focus_end = "none" if manifest.focus_end_s is None else f"{manifest.focus_end_s:.6f}"
    lines = [
        "Video metadata. Times are clip-relative seconds from the start of the supplied file.",
        "Do not invent a different duration, frame rate, or timestamp.",
        f"duration_s: {metadata.duration_s:.6f}",
        f"resolution: {metadata.width}x{metadata.height}",
        f"frame_rate: {metadata.frame_rate:.6f}",
        f"frame_rate_raw: {metadata.frame_rate_raw}",
        f"time_base: {metadata.time_base}",
        f"extracted_frame_count: {len(manifest.frames)}",
        *_coverage_lines(manifest),
        f"broad_fps: {manifest.broad_fps}",
        f"focused_fps: {manifest.focused_fps}",
        f"focus_start_s: {focus_start}",
        f"focus_end_s: {focus_end}",
        "",
        "Frame manifest. time_s is the presentation timestamp of that frame.",
        "frame_id\ttime_s\tsampling\tframe_path\tcontact_sheet",
    ]
    sheet_for = {}
    for sheet in manifest.contact_sheets:
        for frame_id in sheet.frame_ids:
            sheet_for[frame_id] = sheet.sheet_id
    for frame in manifest.frames:
        lines.append(
            f"{frame.frame_id}\t{frame.time_s:.6f}\t{frame.sampling}\t"
            f"{frame.path}\t{sheet_for.get(frame.frame_id, '')}"
        )
    return "\n".join(lines)


def _multi_angle_block(manifest: FrameManifest) -> str:
    lines = [
        "These frames come from separate camera angles of one play.",
        "Each angle has its own clock. A time on one angle is not the same moment on another angle.",
        "Follow the named subject on every angle. If a different player goes to the ground, keep the timeline on the subject.",
        "Do not invent timestamps. Use only the time printed for the cited frame on its own angle.",
        f"extracted_frame_count: {len(manifest.frames)}",
        *_coverage_lines(manifest),
        f"broad_fps: {manifest.broad_fps}",
        f"focused_fps: {manifest.focused_fps}",
        "",
    ]
    for clip in manifest.clips:
        focus = "none"
        if clip.focus_start_s is not None and clip.focus_end_s is not None:
            focus = f"{clip.focus_start_s:.6f}-{clip.focus_end_s:.6f}"
        lines.append(
            f"angle: {clip.clip_id} duration_s: {clip.duration_s:.6f} "
            f"frame_rate: {clip.frame_rate:.6f} focused_window_s: {focus}"
        )
    lines.extend(
        [
            "",
            "Frame manifest. time_s is the presentation timestamp on that angle's own file.",
            "frame_id\tangle\ttime_s\tsampling\tframe_path\tcontact_sheet",
        ]
    )
    sheet_for = {}
    for sheet in manifest.contact_sheets:
        for frame_id in sheet.frame_ids:
            sheet_for[frame_id] = sheet.sheet_id
    for frame in manifest.frames:
        lines.append(
            f"{frame.frame_id}\t{frame.clip_id}\t{frame.time_s:.6f}\t{frame.sampling}\t"
            f"{frame.path}\t{sheet_for.get(frame.frame_id, '')}"
        )
    return "\n".join(lines)


def stage_manifest(
    manifest: FrameManifest,
    output_dir: Path,
    frame_ids: set[str],
    *,
    stage: str,
    max_individual: int,
) -> FrameManifest:
    """A copy of the manifest narrowed to the frames one stage needs.

    The stage's prompt lists only these frames, its images show only these frames, and its
    validator rejects citations of any other frame. When the frames do not fit as individual
    images, new contact sheets are built from them alone, so unrelated frames are not sent.
    """
    directory = f"stage_sheets/{stage}"
    shutil.rmtree(output_dir / directory, ignore_errors=True)
    frames = [frame for frame in manifest.frames if frame.frame_id in frame_ids]
    if not frames:
        raise InjuryAnalysisError(f"No cited frames are available for the {stage} stage")
    if len(frames) == len(manifest.frames):
        return manifest
    subset = manifest.model_copy(
        update={"frames": frames, "contact_sheets": [], "total_frame_count": len(manifest.frames)}
    )
    if len(frames) <= max_individual:
        return subset
    return build_contact_sheets(
        subset,
        output_dir,
        columns=manifest.sheet_columns or 3,
        frames_per_sheet=manifest.frames_per_sheet or 6,
        directory=directory,
        sheet_prefix=f"{stage}_sheet",
    )


def _coverage_lines(manifest: FrameManifest) -> list[str]:
    if manifest.total_frame_count is None:
        return []
    return [
        f"frames_shown_to_this_stage: {len(manifest.frames)} of {manifest.total_frame_count}. "
        "These are the frames cited by the earlier passes. Cite only frames listed below."
    ]


def model_images(manifest: FrameManifest, output_dir: Path, max_individual: int) -> list[ImagePayload]:
    if max_individual < 1:
        raise InjuryAnalysisError("max individual images must be at least 1")
    if len(manifest.frames) <= max_individual:
        payloads = []
        for frame in manifest.frames:
            path = output_dir / frame.path
            if not path.is_file():
                raise InjuryAnalysisError(f"Extracted frame is missing: {frame.path}")
            payloads.append(
                ImagePayload(
                    relative_path=frame.path,
                    path=path,
                    label=(
                    f"Frame {frame.frame_id}"
                    + (f" on angle {frame.clip_id}" if frame.clip_id else "")
                    + f" at clip time {frame.time_s:.6f} seconds. "
                    "This timestamp belongs to that angle only. Do not invent another time."
                    ),
                )
            )
        return payloads
    if not manifest.contact_sheets:
        raise InjuryAnalysisError(
            "Contact sheets are required when individual frames exceed the model image limit"
        )
    payloads = []
    for sheet in manifest.contact_sheets:
        path = output_dir / sheet.path
        if not path.is_file():
            raise InjuryAnalysisError(f"Contact sheet is missing: {sheet.path}")
        payloads.append(
            ImagePayload(
                relative_path=sheet.path,
                path=path,
                label=(
                    f"Contact sheet {sheet.sheet_id}. Each tile is one extracted frame. "
                    f"Frame IDs: {', '.join(sheet.frame_ids)}. "
                    "The time printed on each tile is that angle's own presentation timestamp. "
                    "Angles do not share a clock."
                ),
            )
        )
    return payloads


def json_block(tag: str, model: BaseModel) -> str:
    return f"<{tag}>\n{model.model_dump_json(indent=2)}\n</{tag}>"
