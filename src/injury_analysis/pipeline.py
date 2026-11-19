from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from injury_analysis.config import RunConfig, load_project_env
from injury_analysis.constants import TIMESTAMP_AUTHORITY
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.anthropic_adapter import AnthropicAdapter
from injury_analysis.models.muse_adapter import MuseAdapter
from injury_analysis.models.openai_adapter import OpenAIAdapter
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.model_adapter import (
    ModelAdapter,
    ModelRequest,
    UsageRecorder,
    render_prompt_log,
)
from injury_analysis.models.schemas import (
    CandidateArtifact,
    CandidateReport,
    JudgeArtifact,
    JudgeReport,
    ObserverArtifact,
    ObserverReport,
)
from injury_analysis.models.validation import validate_observer
from injury_analysis.provenance import file_sha256, request_fingerprint
from injury_analysis.stages.blind_reanalysis import run_blind_reanalysis
from injury_analysis.stages.candidate_analyzer import run_candidate_analyzer
from injury_analysis.stages.compare import compare_observations
from injury_analysis.stages.judge import run_judge
from injury_analysis.stages.observer import build_observer_request, run_observer
from injury_analysis.stages.materials import stage_manifest
from injury_analysis.stages.report import render_report
from injury_analysis.video.contact_sheet import build_contact_sheets
from injury_analysis.video.frames import ClipRecord, FrameManifest, extract_frames
from injury_analysis.video.metadata import inspect_video

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_dir: Path
    report_path: Path
    adapter: ModelAdapter


def run_pipeline(config: RunConfig, adapter: ModelAdapter | None = None) -> PipelineResult:
    video_paths = [clip.video_path for clip in config.clips] or [config.video_path]
    for path in video_paths:
        if not path.is_file():
            raise InjuryAnalysisError(f"Video file does not exist: {path}")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    recorder: UsageRecorder | None = None
    started = time.perf_counter()
    _write_run_config(config, status="running")
    try:
        if config.clips:
            metadata, manifest = _inspect_clips(config, output_dir)
        else:
            logger.info("Inspecting %s", config.video_path)
            metadata, probed = inspect_video(
                config.video_path,
                source_offset_s=config.source_offset_s,
                max_duration_s=config.max_duration_s,
            )
            _check_focus_window(config, metadata.duration_s)
            manifest = _frames(config, output_dir, metadata, probed)
            # Written after the resume check so a refused resume leaves the saved metadata alone.
            _write_model(output_dir / "metadata.json", metadata)
        _write_model(output_dir / "frame_manifest.json", manifest)
        logger.info("Using %s extracted frames", len(manifest.frames))

        active = adapter if adapter is not None else build_adapter(config)
        producer, model_name = _identity(config, active)
        recorder = UsageRecorder(active)
        video_name = "\n".join(path.name for path in video_paths)

        def blind_pass():
            # Same frames and observer instructions in a fresh conversation. It never receives
            # the first report or the ranking, so it can run while those are being produced.
            logger.info("Running blind reanalysis")
            return run_blind_reanalysis(
                recorder,
                metadata,
                manifest,
                output_dir,
                video_name=video_name,
                max_individual_images=config.max_individual_images,
                subjects=config.subjects,
            )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="blind") as pool:
            blind_future = pool.submit(blind_pass) if config.parallel_passes else None
            try:
                observer, observer_artifact = _observer_stage(
                    config, metadata, manifest, output_dir, recorder, producer, model_name, video_name, stages
                )

                logger.info("Running candidate analysis")
                candidate_manifest = _stage_manifest(config, manifest, "candidate", _observation_frames(observer))
                candidate, candidate_requests = run_candidate_analyzer(
                    recorder,
                    metadata,
                    candidate_manifest,
                    observer,
                    output_dir,
                    video_name=video_name,
                    max_individual_images=config.max_individual_images,
                    subjects=config.subjects,
                )
                _write_prompts(output_dir, candidate_requests)
                candidate_artifact = _stamp_candidate(
                    candidate,
                    producer,
                    model_name,
                    request_fingerprint(candidate_requests[0], producer, model_name),
                )
                _write_model(output_dir / "candidate_report.json", candidate_artifact)
                stages["candidate"] = "ran"
            except BaseException:
                if blind_future is not None:
                    blind_future.cancel()
                raise
            blind, blind_notes, blind_requests = (
                blind_future.result() if blind_future is not None else blind_pass()
            )
        _write_prompts(output_dir, blind_requests)
        blind_artifact = _stamp_observer(
            blind,
            blind_notes,
            producer,
            model_name,
            request_fingerprint(blind_requests[0], producer, model_name),
        )
        _write_model(output_dir / "blind_reanalysis.json", blind_artifact)
        stages["blind_reanalysis"] = "ran"

        logger.info("Running judge")
        judge_manifest = _stage_manifest(
            config,
            manifest,
            "judge",
            _observation_frames(observer) | _observation_frames(blind) | _candidate_frames(candidate),
        )
        judge, judge_notes, judge_requests = run_judge(
            recorder,
            metadata,
            judge_manifest,
            observer,
            candidate,
            blind,
            output_dir,
            # Full manifest: pairing needs every cited frame's angle and clock.
            comparison=compare_observations(observer, blind, manifest),
            video_name=video_name,
            max_individual_images=config.max_individual_images,
            subjects=config.subjects,
        )
        _write_prompts(output_dir, judge_requests)
        judge_artifact = _stamp_judge(
            judge,
            judge_notes,
            producer,
            model_name,
            request_fingerprint(judge_requests[0], producer, model_name),
        )
        _write_model(output_dir / "judge_report.json", judge_artifact)
        stages["judge"] = "ran"

        report = render_report(
            metadata,
            manifest,
            observer_artifact,
            candidate_artifact,
            blind_artifact,
            judge_artifact,
            subjects=config.subjects,
            stage_frames={
                "candidate": len(candidate_manifest.frames),
                "judge": len(judge_manifest.frames),
            },
        )
        report_path = output_dir / "report.md"
        report_path.write_text(report)
        _write_usage(output_dir, recorder)
        _write_run_config(
            config,
            status="completed",
            stages=stages,
            usage=_usage_totals(recorder, started),
        )
        logger.info("Wrote %s", report_path)
        return PipelineResult(output_dir=output_dir, report_path=report_path, adapter=active)
    except Exception as exc:
        _write_usage(output_dir, recorder)
        _write_run_config(
            config,
            status="failed",
            stages=stages,
            usage=_usage_totals(recorder, started),
            error=str(exc),
        )
        raise


def _observer_stage(config, metadata, manifest, output_dir, recorder, producer, model_name, video_name, stages):
    reused_observer = _reuse_observer(config, metadata, manifest, producer, model_name)
    if reused_observer is not None:
        stages["observer"] = "reused"
        return reused_observer
    logger.info("Running observer")
    observer, observer_notes, observer_requests = run_observer(
        recorder,
        metadata,
        manifest,
        output_dir,
        video_name=video_name,
        max_individual_images=config.max_individual_images,
        subjects=config.subjects,
    )
    _write_prompts(output_dir, observer_requests)
    observer_artifact = _stamp_observer(
        observer,
        observer_notes,
        producer,
        model_name,
        request_fingerprint(observer_requests[0], producer, model_name),
    )
    _write_model(output_dir / "observer_report.json", observer_artifact)
    stages["observer"] = "ran"
    return observer, observer_artifact


def build_adapter(config: RunConfig) -> ModelAdapter:
    if config.mock:
        return MockModelAdapter()
    load_project_env()
    if config.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise InjuryAnalysisError(
                "OPENAI_API_KEY is not set. Add it to .env in the project root, "
                "or re-run with --mock to use the fixture model."
            )
        return OpenAIAdapter(model=config.model_name, api_key=api_key)
    if config.provider == "muse":
        api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY")
        if not api_key:
            raise InjuryAnalysisError(
                "MODEL_API_KEY is not set. Add the Muse Spark key to .env in the project root, "
                "or re-run with --mock to use the fixture model."
            )
        return MuseAdapter(model=config.model_name, api_key=api_key)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise InjuryAnalysisError(
            "ANTHROPIC_API_KEY is not set. Add it to .env in the project root, "
            "export it, or re-run with --mock to use the fixture model."
        )
    return AnthropicAdapter(model=config.model_name, api_key=api_key)


def _stage_manifest(config: RunConfig, manifest: FrameManifest, stage: str, frame_ids: set[str]) -> FrameManifest:
    if config.stage_images == "all":
        frame_ids = {frame.frame_id for frame in manifest.frames}
    narrowed = stage_manifest(
        manifest,
        config.output_dir,
        frame_ids,
        stage=stage,
        max_individual=config.max_individual_images,
    )
    logger.info("The %s stage sees %s of %s frames", stage, len(narrowed.frames), len(manifest.frames))
    return narrowed


def _observation_frames(report: ObserverReport) -> set[str]:
    return {frame_id for observation in report.observations for frame_id in observation.supporting_frame_ids}


def _candidate_frames(report: CandidateReport) -> set[str]:
    frames: set[str] = set()
    for item in report.candidates:
        frames.update(item.frame_ids)
        for link in [*item.supporting_evidence, *item.weakening_evidence]:
            frames.update(link.frame_ids)
    return frames


def _reuse_observer(
    config: RunConfig,
    metadata,
    manifest: FrameManifest,
    producer: str,
    model_name: str,
) -> tuple[ObserverReport, ObserverArtifact] | None:
    path = config.output_dir / "observer_report.json"
    if not config.resume or not path.is_file():
        return None
    artifact = ObserverArtifact.model_validate_json(path.read_text())
    request = build_observer_request(
        metadata,
        manifest,
        config.output_dir,
        max_individual_images=config.max_individual_images,
        subjects=config.subjects,
    )
    if artifact.input_fingerprint != request_fingerprint(request, producer, model_name):
        logger.info(
            "Saved observer report came from different frames, prompt, subject, or model; "
            "running the observer again"
        )
        return None
    data = artifact.model_dump()
    for key in ("producer", "model_name", "timestamp_authority", "timestamp_adjustments", "input_fingerprint"):
        data.pop(key, None)
    report = ObserverReport.model_validate(data)
    errors = validate_observer(report, manifest)
    if errors:
        logger.info("Saved observer report did not validate; running the observer again")
        return None
    logger.info("Reusing saved observer report")
    return report, artifact


def _inspect_clips(config: RunConfig, output_dir: Path):
    if config.resume:
        manifest_path = output_dir / "frame_manifest.json"
        if not manifest_path.is_file():
            raise InjuryAnalysisError("--resume requires an existing frame_manifest.json in the output directory")
        manifest = FrameManifest.model_validate_json(manifest_path.read_text())
        _require_same_clips(manifest, config)
        _require_frame_files(manifest, output_dir)
        manifest = _ensure_contact_sheets(manifest, config, output_dir)
        saved = json.loads((output_dir / "metadata.json").read_text())
        first = saved[0] if isinstance(saved, list) else saved
        from injury_analysis.video.metadata import VideoMetadata

        logger.info("Reusing extracted frames from %s angles", len(manifest.clips))
        return VideoMetadata.model_validate(first), manifest

    from injury_analysis.video.metadata import VideoMetadata

    records = []
    clip_rows: list[ClipRecord] = []
    metadatas: list[VideoMetadata] = []
    start_number = 1
    remaining = config.max_frames
    for index, clip in enumerate(config.clips):
        logger.info("Inspecting %s (%s)", clip.label, clip.video_path)
        metadata, probed = inspect_video(
            clip.video_path,
            source_offset_s=config.source_offset_s,
            max_duration_s=config.max_duration_s,
        )
        if clip.focus_start_s is not None and clip.focus_end_s is not None:
            if clip.focus_start_s < 0 or clip.focus_end_s > metadata.duration_s + 1e-3:
                raise InjuryAnalysisError(
                    f"{clip.label} window {clip.focus_start_s:.3f}–{clip.focus_end_s:.3f}s "
                    f"is outside the clip duration of {metadata.duration_s:.3f}s."
                )
        if remaining < 1:
            raise InjuryAnalysisError("max_frames was used up before every angle was sampled")
        piece = extract_frames(
            clip.video_path,
            output_dir,
            metadata,
            probed,
            broad_fps=config.broad_fps,
            focused_fps=config.focused_fps,
            focus_start_s=clip.focus_start_s,
            focus_end_s=clip.focus_end_s,
            max_frames=remaining,
            source_offset_s=config.source_offset_s,
            clip_id=clip.label,
            start_number=start_number,
            reset_dir=index == 0,
        )
        records.extend(piece.frames)
        start_number += len(piece.frames)
        remaining -= len(piece.frames)
        clip_rows.append(
            ClipRecord(
                clip_id=clip.label,
                video_path=str(clip.video_path),
                duration_s=metadata.duration_s,
                frame_rate=metadata.frame_rate,
                width=metadata.width,
                height=metadata.height,
                focus_start_s=clip.focus_start_s,
                focus_end_s=clip.focus_end_s,
                video_sha256=file_sha256(clip.video_path),
            )
        )
        metadatas.append(metadata)
    manifest = build_contact_sheets(
        FrameManifest(
            broad_fps=config.broad_fps,
            focused_fps=config.focused_fps,
            frames=records,
            clips=clip_rows,
        ),
        output_dir,
        columns=config.sheet_columns,
        frames_per_sheet=config.frames_per_sheet,
    )
    payload = [json.loads(item.model_dump_json()) for item in metadatas]
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    return metadatas[0], manifest


def _require_same_clips(manifest: FrameManifest, config: RunConfig) -> None:
    saved = [
        (clip.clip_id, clip.focus_start_s, clip.focus_end_s)
        for clip in manifest.clips
    ]
    requested = [
        (clip.label, clip.focus_start_s, clip.focus_end_s)
        for clip in config.clips
    ]
    same = (
        _close(manifest.broad_fps, config.broad_fps)
        and _close(manifest.focused_fps, config.focused_fps)
        and len(saved) == len(requested)
        and all(
            left[0] == right[0] and _close(left[1], right[1]) and _close(left[2], right[2])
            for left, right in zip(saved, requested)
        )
    )
    if not same:
        raise InjuryAnalysisError(
            "Saved frames were extracted with different angles or sampling. Re-run without --resume."
        )
    for saved_clip, clip in zip(manifest.clips, config.clips, strict=True):
        _require_same_video(saved_clip.video_sha256, clip.video_path, f"Angle {clip.label}")


def _frames(config: RunConfig, output_dir: Path, metadata, probed) -> FrameManifest:
    manifest_path = output_dir / "frame_manifest.json"
    if config.resume:
        if not manifest_path.is_file():
            raise InjuryAnalysisError("--resume requires an existing frame_manifest.json in the output directory")
        manifest = FrameManifest.model_validate_json(manifest_path.read_text())
        _require_same_sampling(manifest, config)
        _require_same_video(manifest.video_sha256, config.video_path, "The video")
        _require_frame_files(manifest, output_dir)
        logger.info("Reusing extracted frames")
        return _ensure_contact_sheets(manifest, config, output_dir)

    manifest = extract_frames(
        config.video_path,
        output_dir,
        metadata,
        probed,
        broad_fps=config.broad_fps,
        focused_fps=config.focused_fps,
        focus_start_s=config.focus_start_s,
        focus_end_s=config.focus_end_s,
        max_frames=config.max_frames,
        source_offset_s=config.source_offset_s,
    )
    manifest = manifest.model_copy(update={"video_sha256": file_sha256(config.video_path)})
    return build_contact_sheets(
        manifest,
        output_dir,
        columns=config.sheet_columns,
        frames_per_sheet=config.frames_per_sheet,
    )


def _require_same_video(saved_sha256: str | None, video_path: Path, label: str) -> None:
    if saved_sha256 is None:
        raise InjuryAnalysisError(
            "Saved frames have no video fingerprint because an older version extracted them. "
            "Re-run without --resume."
        )
    if saved_sha256 != file_sha256(video_path):
        raise InjuryAnalysisError(
            f"{label} is not the file the saved frames came from. Re-run without --resume, "
            "or use a new --output directory."
        )


def _require_frame_files(manifest: FrameManifest, output_dir: Path) -> None:
    missing = [frame.path for frame in manifest.frames if not (output_dir / frame.path).is_file()]
    if missing:
        raise InjuryAnalysisError(
            "Cannot resume because extracted frames are missing: " + ", ".join(missing[:5])
        )


def _ensure_contact_sheets(manifest: FrameManifest, config: RunConfig, output_dir: Path) -> FrameManifest:
    current = (
        bool(manifest.contact_sheets)
        and manifest.sheet_columns == config.sheet_columns
        and manifest.frames_per_sheet == config.frames_per_sheet
        and all((output_dir / sheet.path).is_file() for sheet in manifest.contact_sheets)
    )
    if current:
        return manifest
    return build_contact_sheets(
        manifest,
        output_dir,
        columns=config.sheet_columns,
        frames_per_sheet=config.frames_per_sheet,
    )


def _require_same_sampling(manifest: FrameManifest, config: RunConfig) -> None:
    same = (
        _close(manifest.broad_fps, config.broad_fps)
        and _close(manifest.focused_fps, config.focused_fps)
        and _close(manifest.focus_start_s, config.focus_start_s)
        and _close(manifest.focus_end_s, config.focus_end_s)
        and _close(manifest.source_offset_s, config.source_offset_s)
    )
    if not same:
        raise InjuryAnalysisError(
            "Saved frames were extracted with different sampling settings. Re-run without --resume."
        )


def _check_focus_window(config: RunConfig, duration_s: float) -> None:
    if config.focus_start_s is None or config.focus_end_s is None:
        return
    if config.focus_start_s < 0 or config.focus_end_s > duration_s + 1e-3:
        raise InjuryAnalysisError(
            f"Focused window {config.focus_start_s:.3f}–{config.focus_end_s:.3f}s "
            f"is outside the clip duration of {duration_s:.3f}s."
        )


def _identity(config: RunConfig, adapter: ModelAdapter) -> tuple[str, str]:
    if config.mock or isinstance(adapter, MockModelAdapter):
        return "mock", "mock-fixture"
    return config.provider, config.model_name


def _stamp_observer(
    report: ObserverReport,
    notes: list[str],
    producer: str,
    model_name: str,
    fingerprint: str,
) -> ObserverArtifact:
    return ObserverArtifact(
        producer=producer,  # type: ignore[arg-type]
        model_name=model_name,
        timestamp_authority=TIMESTAMP_AUTHORITY,
        timestamp_adjustments=notes,
        input_fingerprint=fingerprint,
        **report.model_dump(),
    )


def _stamp_candidate(report: CandidateReport, producer: str, model_name: str, fingerprint: str) -> CandidateArtifact:
    return CandidateArtifact(
        producer=producer,  # type: ignore[arg-type]
        model_name=model_name,
        timestamp_authority=TIMESTAMP_AUTHORITY,
        input_fingerprint=fingerprint,
        **report.model_dump(),
    )


def _stamp_judge(
    report: JudgeReport,
    notes: list[str],
    producer: str,
    model_name: str,
    fingerprint: str,
) -> JudgeArtifact:
    return JudgeArtifact(
        producer=producer,  # type: ignore[arg-type]
        model_name=model_name,
        timestamp_authority=TIMESTAMP_AUTHORITY,
        timestamp_adjustments=notes,
        input_fingerprint=fingerprint,
        **report.model_dump(),
    )


def _write_prompts(output_dir: Path, requests: list[ModelRequest]) -> None:
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    stage = requests[0].stage
    # Drop logs from an earlier invocation so every file matches the saved artifact.
    for stale in [prompt_dir / f"{stage}.txt", *prompt_dir.glob(f"{stage}_retry*.txt")]:
        stale.unlink(missing_ok=True)
    for index, request in enumerate(requests):
        suffix = "" if index == 0 else f"_retry{index}"
        (prompt_dir / f"{request.stage}{suffix}.txt").write_text(render_prompt_log(request))


def _write_model(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n")


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) < 1e-6


def _write_run_config(
    config: RunConfig,
    *,
    status: str,
    stages: dict[str, str] | None = None,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    payload = config.model_dump(mode="json")
    payload["status"] = status
    payload["producer"] = "mock" if config.mock else config.provider
    (config.output_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    if status == "running":
        return
    # run_config.json holds only the latest invocation. This log keeps every one.
    record = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "stages": stages or {},
        "usage": usage or {},
        "config": payload,
    }
    if error:
        record["error"] = error[:2000]
    with (config.output_dir / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_usage(output_dir: Path, recorder: UsageRecorder | None) -> None:
    """One row per model call in this invocation.

    An invocation that made no model calls, such as a refused resume, leaves the previous file
    in place, because the saved artifacts still come from that earlier invocation.
    """
    rows = recorder.records if recorder is not None else []
    if not rows:
        return
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (output_dir / "usage.jsonl").write_text(text)


def _usage_totals(recorder: UsageRecorder | None, started: float) -> dict:
    rows = recorder.records if recorder is not None else []

    def total(key: str) -> int | None:
        values = [row[key] for row in rows]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    return {
        "model_calls": len(rows),
        "images_sent": sum(row["images"] for row in rows),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "model_latency_s": round(sum(row["latency_s"] for row in rows), 3),
        "wall_s": round(time.perf_counter() - started, 3),
    }
