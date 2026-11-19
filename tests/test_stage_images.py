import pytest

from injury_analysis.config import RunConfig
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.schemas import CandidateReport
from injury_analysis.models.validation import validate_candidate
from injury_analysis.pipeline import run_pipeline
from injury_analysis.video.frames import FrameManifest


class PartialObserver(MockModelAdapter):
    """Both observation passes cite only their first two phase groups, like a real model that skips frames."""

    def complete(self, request):
        report = super().complete(request)
        if request.stage in ("observer", "blind_reanalysis"):
            return report.model_copy(update={"observations": report.observations[:2]})
        return report


def _config(video, output, **overrides) -> RunConfig:
    payload = dict(
        video_path=video,
        output_dir=output,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.4,
        focus_end_s=1.2,
        mock=True,
        max_individual_images=40,
    )
    payload.update(overrides)
    return RunConfig(**payload)


def _cited(report) -> set[str]:
    return {frame_id for observation in report.observations for frame_id in observation.supporting_frame_ids}


def _call(adapter, stage):
    return next(call for call in adapter.calls if call.stage == stage)


def test_candidate_and_judge_see_only_cited_frames(synthetic_video, tmp_path):
    output = tmp_path / "run"
    adapter = PartialObserver()
    run_pipeline(_config(synthetic_video, output), adapter=adapter)

    manifest = FrameManifest.model_validate_json((output / "frame_manifest.json").read_text())
    total = len(manifest.frames)
    observer, candidate, blind, judge = (
        _call(adapter, stage) for stage in ("observer", "candidate", "blind_reanalysis", "judge")
    )

    assert len(observer.images) == total and len(blind.images) == total
    assert observer.user_text == blind.user_text

    observer_cited = {
        frame_id
        for observation in PartialObserver().complete(observer).observations
        for frame_id in observation.supporting_frame_ids
    }
    assert {frame.frame_id for frame in candidate.manifest.frames} == observer_cited
    assert len(candidate.images) == len(observer_cited) < total
    assert f"frames_shown_to_this_stage: {len(observer_cited)} of {total}" in candidate.user_text
    unseen = sorted({frame.frame_id for frame in manifest.frames} - observer_cited)
    assert all(f"\n{frame_id}\t" not in candidate.user_text for frame_id in unseen)

    judge_frames = {frame.frame_id for frame in judge.manifest.frames}
    assert observer_cited <= judge_frames < {frame.frame_id for frame in manifest.frames}
    assert len(judge.images) == len(judge_frames)

    report = (output / "report.md").read_text()
    assert f"The candidate stage saw {len(observer_cited)} of {total} extracted frames" in report


def test_stage_images_all_keeps_the_old_behaviour(synthetic_video, tmp_path):
    adapter = PartialObserver()
    run_pipeline(_config(synthetic_video, tmp_path / "run", stage_images="all"), adapter=adapter)

    observer = _call(adapter, "observer")
    for stage in ("candidate", "judge"):
        call = _call(adapter, stage)
        assert len(call.images) == len(observer.images)
        assert "frames_shown_to_this_stage" not in call.user_text


def test_cited_frames_are_packed_into_new_sheets_when_they_do_not_fit(synthetic_video, tmp_path):
    output = tmp_path / "run"
    adapter = PartialObserver()
    run_pipeline(_config(synthetic_video, output, max_individual_images=2), adapter=adapter)

    candidate = _call(adapter, "candidate")
    shown = {frame.frame_id for frame in candidate.manifest.frames}
    assert candidate.images
    assert all(image.relative_path.startswith("stage_sheets/candidate/candidate_sheet_") for image in candidate.images)
    assert all(image.path.is_file() for image in candidate.images)
    packed = {frame_id for sheet in candidate.manifest.contact_sheets for frame_id in sheet.frame_ids}
    assert packed == shown
    main_sheets = FrameManifest.model_validate_json((output / "frame_manifest.json").read_text()).contact_sheets
    assert len(candidate.images) < len(main_sheets)


def test_candidate_may_not_cite_a_frame_it_was_not_shown(synthetic_video, tmp_path):
    adapter = PartialObserver()
    run_pipeline(_config(synthetic_video, tmp_path / "run"), adapter=adapter)
    candidate = _call(adapter, "candidate")
    shown = [frame.frame_id for frame in candidate.manifest.frames]
    full = FrameManifest.model_validate_json((tmp_path / "run" / "frame_manifest.json").read_text())
    unseen = next(frame.frame_id for frame in full.frames if frame.frame_id not in shown)

    report = MockModelAdapter().complete(candidate)
    assert isinstance(report, CandidateReport)
    observer = PartialObserver().complete(_call(adapter, "observer"))
    tampered = report.model_copy(
        update={"candidates": [report.candidates[0].model_copy(update={"frame_ids": [unseen]}), *report.candidates[1:]]}
    )
    errors = validate_candidate(tampered, candidate.manifest, observer)
    assert any(unseen in error for error in errors)


def test_narrowing_does_not_change_observer_reuse(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output), adapter=PartialObserver())
    again = PartialObserver()
    run_pipeline(_config(synthetic_video, output, resume=True), adapter=again)
    assert [call.stage for call in again.calls] == ["candidate", "blind_reanalysis", "judge"]


def test_stage_images_must_be_a_known_mode(synthetic_video, tmp_path):
    with pytest.raises(ValueError):
        _config(synthetic_video, tmp_path / "run", stage_images="some")
