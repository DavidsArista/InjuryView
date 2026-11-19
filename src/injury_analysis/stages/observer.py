from __future__ import annotations

from pathlib import Path

from injury_analysis.constants import STAGE_OBSERVER
from injury_analysis.models.model_adapter import ModelAdapter, ModelRequest, complete_with_retry
from injury_analysis.models.schemas import ObserverReport
from injury_analysis.models.validation import bind_observer, validate_observer
from injury_analysis.stages.isolation import assert_safe_prompt
from injury_analysis.stages.materials import metadata_block, model_images, subject_block
from injury_analysis.stages.prompts import OBSERVER_SYSTEM
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def run_observer(
    adapter: ModelAdapter,
    metadata: VideoMetadata,
    manifest: FrameManifest,
    output_dir: Path,
    *,
    video_name: str,
    max_individual_images: int,
    subjects: list[str] | None = None,
) -> tuple[ObserverReport, list[str], list[ModelRequest]]:
    request = build_observer_request(
        metadata,
        manifest,
        output_dir,
        max_individual_images=max_individual_images,
        subjects=subjects,
    )
    assert_safe_prompt(request.stage, request.system, request.user_text, video_name)
    report, requests = complete_with_retry(
        adapter,
        request,
        lambda item: validate_observer(item, manifest),
    )
    bound, notes = bind_observer(report, manifest)
    return bound, notes, requests


def build_observer_request(
    metadata: VideoMetadata,
    manifest: FrameManifest,
    output_dir: Path,
    *,
    max_individual_images: int,
    subjects: list[str] | None = None,
) -> ModelRequest:
    user_text = (
        subject_block(subjects or [])
        + metadata_block(metadata, manifest)
        + "\n\nDescribe only what is visible in these frames. "
        "Cover every phase the frames actually support, and list the phases you cannot see. "
        "When the view is limited, say unclear.\n"
    )
    return ModelRequest(
        stage=STAGE_OBSERVER,
        system=OBSERVER_SYSTEM,
        user_text=user_text,
        images=model_images(manifest, output_dir, max_individual_images),
        response_model=ObserverReport,
        manifest=manifest,
    )
