from __future__ import annotations

from pathlib import Path

from injury_analysis.constants import OBSERVER_BLOCK, STAGE_CANDIDATE
from injury_analysis.models.model_adapter import ModelAdapter, ModelRequest, complete_with_retry
from injury_analysis.models.schemas import CandidateReport, ObserverReport
from injury_analysis.models.validation import normalize_candidate, validate_candidate
from injury_analysis.stages.isolation import assert_safe_prompt
from injury_analysis.stages.materials import json_block, metadata_block, model_images, subject_block
from injury_analysis.stages.prompts import CANDIDATE_SYSTEM
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def run_candidate_analyzer(
    adapter: ModelAdapter,
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverReport,
    output_dir: Path,
    *,
    video_name: str,
    max_individual_images: int,
    subjects: list[str] | None = None,
) -> tuple[CandidateReport, list[ModelRequest]]:
    request = ModelRequest(
        stage=STAGE_CANDIDATE,
        system=CANDIDATE_SYSTEM,
        user_text=_user_text(metadata, manifest, observer, subjects or []),
        images=model_images(manifest, output_dir, max_individual_images),
        response_model=CandidateReport,
        manifest=manifest,
    )
    assert_safe_prompt(request.stage, request.system, request.user_text, video_name)
    report, requests = complete_with_retry(
        adapter,
        request,
        lambda item: validate_candidate(item, manifest, observer),
        normalize=lambda item: normalize_candidate(item, observer),
    )
    return report, requests


def _user_text(
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverReport,
    subjects: list[str],
) -> str:
    return (
        subject_block(subjects)
        + metadata_block(metadata, manifest)
        + "\n\nObserver timeline. Timestamps in this block were bound to frame presentation timestamps.\n"
        + json_block(OBSERVER_BLOCK, observer)
        + "\n\nRank plausible visible-mechanism patterns from this timeline and the frames. "
        "Include exactly one insufficient_visual_evidence candidate. "
        "Do not assume which pattern should win.\n"
    )
