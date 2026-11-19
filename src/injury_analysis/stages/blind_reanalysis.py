from __future__ import annotations

from pathlib import Path

from injury_analysis.constants import STAGE_BLIND
from injury_analysis.models.model_adapter import ModelAdapter, ModelRequest, complete_with_retry
from injury_analysis.models.schemas import ObserverReport
from injury_analysis.models.validation import bind_observer, validate_observer
from injury_analysis.stages.isolation import assert_safe_prompt
from injury_analysis.stages.observer import build_observer_request
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def run_blind_reanalysis(
    adapter: ModelAdapter,
    metadata: VideoMetadata,
    manifest: FrameManifest,
    output_dir: Path,
    *,
    video_name: str,
    max_individual_images: int,
    subjects: list[str] | None = None,
) -> tuple[ObserverReport, list[str], list[ModelRequest]]:
    """Second observer pass. The request matches the first observer, including who to watch."""
    observer_request = build_observer_request(
        metadata,
        manifest,
        output_dir,
        max_individual_images=max_individual_images,
        subjects=subjects,
    )
    request = ModelRequest(
        stage=STAGE_BLIND,
        system=observer_request.system,
        user_text=observer_request.user_text,
        images=observer_request.images,
        response_model=ObserverReport,
        manifest=manifest,
    )
    assert_safe_prompt(request.stage, request.system, request.user_text, video_name)
    report, requests = complete_with_retry(
        adapter,
        request,
        lambda item: validate_observer(item, manifest),
    )
    bound, notes = bind_observer(report, manifest)
    return bound, notes, requests
