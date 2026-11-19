from __future__ import annotations

from pathlib import Path

from injury_analysis.constants import BLIND_BLOCK, CANDIDATE_BLOCK, OBSERVER_BLOCK, STAGE_JUDGE
from injury_analysis.models.model_adapter import ModelAdapter, ModelRequest, complete_with_retry
from injury_analysis.models.schemas import CandidateReport, JudgeReport, ObserverReport
from injury_analysis.models.validation import bind_judge, validate_judge
from injury_analysis.stages.isolation import assert_safe_prompt
from injury_analysis.stages.materials import json_block, metadata_block, model_images, subject_block
from injury_analysis.stages.prompts import JUDGE_SYSTEM
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def run_judge(
    adapter: ModelAdapter,
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverReport,
    candidate: CandidateReport,
    blind: ObserverReport,
    output_dir: Path,
    *,
    video_name: str,
    max_individual_images: int,
    subjects: list[str] | None = None,
) -> tuple[JudgeReport, list[str], list[ModelRequest]]:
    request = ModelRequest(
        stage=STAGE_JUDGE,
        system=JUDGE_SYSTEM,
        user_text=_user_text(metadata, manifest, observer, candidate, blind, subjects or []),
        images=model_images(manifest, output_dir, max_individual_images),
        response_model=JudgeReport,
        manifest=manifest,
    )
    assert_safe_prompt(request.stage, request.system, request.user_text, video_name)
    report, requests = complete_with_retry(
        adapter,
        request,
        lambda item: validate_judge(item, manifest),
    )
    bound, notes = bind_judge(report, manifest)
    return bound, notes, requests


def _user_text(
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverReport,
    candidate: CandidateReport,
    blind: ObserverReport,
    subjects: list[str],
) -> str:
    return (
        subject_block(subjects)
        + metadata_block(metadata, manifest)
        + "\n\nFirst observer timeline:\n"
        + json_block(OBSERVER_BLOCK, observer)
        + "\n\nCandidate report:\n"
        + json_block(CANDIDATE_BLOCK, candidate)
        + "\n\nBlind second observer timeline. This pass did not see the first timeline or the candidate report:\n"
        + json_block(BLIND_BLOCK, blind)
        + "\n\nCompare the claims with the frames and with both observation passes. "
        "Expose disagreement. Do not try to make the two runs agree.\n"
    )
