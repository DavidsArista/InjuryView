import subprocess

import pytest
from PIL import Image, ImageChops

from injury_analysis.config import ClipSpec, RunConfig
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.schemas import (
    BodySide,
    ClaimAssessment,
    ClaimVerdict,
    Confidence,
    ContactStatus,
    JudgeReport,
    Observation,
    ObserverArtifact,
    ObserverReport,
    Phase,
    RankingDecision,
    StatementKind,
)
from injury_analysis.models.validation import authoritative_span, validate_judge
from injury_analysis.pipeline import run_pipeline
from injury_analysis.stages.compare import compare_observations
from injury_analysis.video.frames import FrameManifest, FrameRecord, extract_frames
from injury_analysis.video.metadata import inspect_video
from injury_analysis.video.synthetic import write_synthetic_video


def _manifest() -> FrameManifest:
    def frame(frame_id: str, angle: str, time_s: float) -> FrameRecord:
        return FrameRecord(
            frame_id=frame_id,
            source_frame_index=0,
            pts_s=time_s,
            time_s=time_s,
            path=f"frames/{frame_id}.jpg",
            width=8,
            height=8,
            sampling="broad",
            clip_id=angle,
        )

    return FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            frame("f001", "wide", 5.0),
            frame("f002", "wide", 6.0),
            frame("f003", "tight", 5.0),
            frame("f004", "tight", 6.0),
        ],
    )


def _observation(observation_id: str, frames: list[str], *, player="QB 18", side=BodySide.RIGHT, start=5.0, end=6.0):
    return Observation(
        observation_id=observation_id,
        start_time_s=start,
        end_time_s=end,
        phase=Phase.ONSET,
        player=player,
        body_region="lower limb",
        body_side=side,
        visible_movement="The stride changes.",
        contact_status=ContactStatus.UNCLEAR,
        supporting_frame_ids=frames,
        confidence=Confidence.LOW,
        statement_kind=StatementKind.OBSERVED,
    )


def _report(*observations: Observation) -> ObserverReport:
    return ObserverReport(summary="Visible movement only.", observations=list(observations))


def test_comparison_never_pairs_observations_from_different_angles():
    manifest = _manifest()
    first = _report(_observation("o1", ["f001", "f002"], side=BodySide.RIGHT))
    second = _report(_observation("o1", ["f003", "f004"], side=BodySide.LEFT))

    comparison = compare_observations(first, second, manifest)

    assert comparison.body_side_disagreements == []
    assert len(comparison.only_in_first) == 1 and "angle=wide" in comparison.only_in_first[0]
    assert len(comparison.only_in_second) == 1 and "angle=tight" in comparison.only_in_second[0]


def test_comparison_never_pairs_two_different_named_players():
    manifest = _manifest()
    first = _report(_observation("o1", ["f001"], player="QB, 18, brown jersey", side=BodySide.RIGHT))
    second = _report(_observation("o1", ["f001"], player="Defender, white jersey", side=BodySide.LEFT))

    comparison = compare_observations(first, second, manifest)

    assert comparison.body_side_disagreements == []
    assert comparison.only_in_first and comparison.only_in_second


def test_unclear_player_pairs_tentatively_and_the_largest_overlap_wins():
    manifest = _manifest()
    first = _report(_observation("o1", ["f001", "f002"], player="QB, 18", start=5.0, end=6.0))
    second = _report(
        _observation("short", ["f001"], player="unclear", start=5.9, end=6.0),
        _observation("long", ["f001", "f002"], player="qb 18", start=5.0, end=6.0),
    )

    comparison = compare_observations(first, second, manifest)

    assert comparison.tentative_pairs == []
    assert len(comparison.only_in_second) == 1 and comparison.only_in_second[0].startswith("short")

    tentative = compare_observations(first, _report(_observation("x", ["f001"], player="unclear")), manifest)
    assert tentative.tentative_pairs and "did not name the player" in tentative.tentative_pairs[0]


def test_judge_claim_spanning_two_angles_is_rejected_and_never_keeps_model_times():
    manifest = _manifest()
    claim = ClaimAssessment(
        claim_id="c1",
        claim_text="The stride changes.",
        verdict=ClaimVerdict.DIRECTLY_SUPPORTED,
        rationale="Visible on both angles.",
        frame_ids=["f001", "f003"],
        start_time_s=123.0,
        end_time_s=456.0,
    )
    report = JudgeReport(
        claim_assessments=[claim],
        ranking_decision=RankingDecision.INCONCLUSIVE,
        ranking_rationale="Unresolved.",
    )

    errors = validate_judge(report, manifest)

    assert any("more than one angle" in error for error in errors)
    with pytest.raises(InjuryAnalysisError, match="do not share a clock"):
        authoritative_span(123.0, 456.0, ["f001", "f003"], manifest)


def test_frames_match_their_timestamps_when_the_container_starts_after_zero(tmp_path):
    source = tmp_path / "source.mp4"
    write_synthetic_video(source, duration_s=3, fps="30", size="160x120")
    shifted = tmp_path / "shifted.ts"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c", "copy", "-output_ts_offset", "5", str(shifted)],
        check=True,
    )
    metadata, probed = inspect_video(shifted)
    assert metadata.container_start_s > 5

    manifest = extract_frames(
        shifted,
        tmp_path / "run",
        metadata,
        probed,
        broad_fps=2,
        focused_fps=12,
        focus_start_s=None,
        focus_end_s=None,
        max_frames=80,
        source_offset_s=None,
    )

    assert manifest.frames[0].time_s == pytest.approx(0.0, abs=1e-6)
    for record in manifest.frames:
        # Same JPEG settings as extraction, so the right frame differs by ~0 and a neighbor by ~0.4.
        reference = tmp_path / f"reference_{record.frame_id}.jpg"
        window = f"between(t\\,{record.time_s - 0.01:.4f}\\,{record.time_s + 0.01:.4f})"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-filter:v", f"select='{window}'",
             "-fps_mode", "vfr", "-frames:v", "1", "-q:v", "2", str(reference)],
            check=True,
        )
        extracted = Image.open(tmp_path / "run" / record.path).convert("L")
        expected = Image.open(reference).convert("L")
        histogram = ImageChops.difference(extracted, expected).histogram()
        mean_difference = sum(value * count for value, count in enumerate(histogram)) / (extracted.width * extracted.height)
        assert mean_difference < 0.05, f"{record.frame_id} at {record.time_s:.3f}s differs by {mean_difference:.2f}"


def test_mock_pipeline_keeps_every_observation_on_one_angle(tmp_path):
    videos = []
    for name, size in (("wide", "320x240"), ("tight", "160x120"), ("end_zone", "240x180")):
        path = tmp_path / f"{name}.mp4"
        write_synthetic_video(path, duration_s=2, size=size)
        videos.append((name, path))
    output = tmp_path / "run"
    config = RunConfig(
        video_path=videos[0][1],
        output_dir=output,
        broad_fps=4,
        focused_fps=12,
        mock=True,
        max_individual_images=6,
        clips=[
            ClipSpec(video_path=path, label=label, focus_start_s=0.5, focus_end_s=1.0)
            for label, path in videos
        ],
    )

    run_pipeline(config, adapter=MockModelAdapter())

    manifest = FrameManifest.model_validate_json((output / "frame_manifest.json").read_text())
    assert all(clip.video_sha256 for clip in manifest.clips)
    for name in ("observer_report.json", "blind_reanalysis.json"):
        report = ObserverArtifact.model_validate_json((output / name).read_text())
        for observation in report.observations:
            angles = {manifest.frame(frame_id).clip_id for frame_id in observation.supporting_frame_ids}
            assert len(angles) == 1, f"{name} {observation.observation_id} spans {angles}"
    report = (output / "report.md").read_text()
    assert "pairs observations only within one camera angle" in report
