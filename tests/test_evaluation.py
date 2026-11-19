import json
import sys
from pathlib import Path

import pytest

from injury_analysis.cli import main
from injury_analysis.config import RunConfig
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.evaluation import AnnotationSet, score_run
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.schemas import (
    BodySide,
    Candidate,
    CandidateArtifact,
    ClaimAssessment,
    ClaimVerdict,
    Confidence,
    ContactStatus,
    JudgeArtifact,
    MechanismCategory,
    Observation,
    ObserverArtifact,
    Phase,
    RankingDecision,
    StatementKind,
)
from injury_analysis.pipeline import run_pipeline
from injury_analysis.provenance import file_sha256
from injury_analysis.video.frames import FrameManifest, FrameRecord

VIDEO_SHA = "a" * 64


def _write(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=2))


def _observation(observation_id, frame_id, time_s, *, phase=Phase.ONSET, side, contact, player="QB 18"):
    return Observation(
        observation_id=observation_id,
        start_time_s=time_s,
        end_time_s=time_s,
        phase=phase,
        player=player,
        body_region="lower limb",
        body_side=side,
        visible_movement="The stride changes.",
        contact_status=contact,
        supporting_frame_ids=[frame_id],
        confidence=Confidence.LOW,
        statement_kind=StatementKind.OBSERVED,
    )


def _observer(*observations):
    return ObserverArtifact(
        producer="mock",
        model_name="mock-fixture",
        timestamp_authority="video_presentation_timestamp",
        summary="Hand-written fixture.",
        observations=list(observations),
    )


@pytest.fixture
def hand_run(tmp_path) -> Path:
    """A run directory whose scores can be worked out by hand."""
    run = tmp_path / "hand-run"
    run.mkdir()
    frames = [
        FrameRecord(
            frame_id=f"f00{index + 1}",
            source_frame_index=index * 8,
            pts_s=index * 0.25,
            time_s=index * 0.25,
            path=f"frames/f00{index + 1}.jpg",
            width=8,
            height=8,
            sampling="broad",
        )
        for index in range(4)
    ]
    _write(run / "frame_manifest.json", FrameManifest(broad_fps=4, focused_fps=12, video_sha256=VIDEO_SHA, frames=frames))
    (run / "metadata.json").write_text(json.dumps({"frame_rate": 30.0}))
    (run / "run_config.json").write_text(json.dumps({"subjects": ["QB, 18"]}))
    _write(
        run / "observer_report.json",
        _observer(
            # A defender's earlier onset must be ignored. The subject's onset is at 0.50 s.
            _observation("o0", "f001", 0.0, side=BodySide.RIGHT, contact=ContactStatus.DIRECT, player="Defender, white"),
            _observation("o1", "f003", 0.5, side=BodySide.LEFT, contact=ContactStatus.DIRECT),
        ),
    )
    _write(
        run / "blind_reanalysis.json",
        _observer(_observation("o1", "f002", 0.25, side=BodySide.RIGHT, contact=ContactStatus.UNCLEAR, player="unclear")),
    )
    _write(
        run / "candidate_report.json",
        CandidateArtifact(
            producer="mock",
            model_name="mock-fixture",
            timestamp_authority="video_presentation_timestamp",
            summary="Fixture ranking.",
            candidates=[
                Candidate(
                    candidate_id="k1",
                    name="Knee pattern",
                    category=MechanismCategory.KNEE,
                    mechanism_description="A knee pattern.",
                    distinguishing_evidence_needed=["Another angle."],
                    confidence=Confidence.LOW,
                    observation_ids=["o1"],
                    frame_ids=["f003"],
                )
            ],
        ),
    )
    _write(
        run / "judge_report.json",
        JudgeArtifact(
            producer="mock",
            model_name="mock-fixture",
            timestamp_authority="video_presentation_timestamp",
            claim_assessments=[
                ClaimAssessment(
                    claim_id="c1",
                    claim_text="The knee changes.",
                    verdict=ClaimVerdict.PARTIALLY_SUPPORTED,
                    rationale="Seen in f003 and again in f004.",
                    frame_ids=["f003"],
                    start_time_s=0.5,
                    end_time_s=0.5,
                ),
                ClaimAssessment(
                    claim_id="c2",
                    claim_text="The stride shortens.",
                    verdict=ClaimVerdict.DIRECTLY_SUPPORTED,
                    rationale="Seen in f002.",
                    frame_ids=["f002"],
                    start_time_s=0.25,
                    end_time_s=0.25,
                ),
            ],
            ranking_decision=RankingDecision.RETAIN,
            ranking_rationale="Fixture.",
        ),
    )
    (run / "usage.jsonl").write_text(
        "".join(
            json.dumps({"stage": stage, "attempt": 1, "images": 4, "input_tokens": 1000, "output_tokens": 200, "latency_s": 1.5, "error": None})
            + "\n"
            for stage in ("observer", "candidate", "blind_reanalysis", "judge")
        )
    )
    return run


def _labels(sha=VIDEO_SHA) -> AnnotationSet:
    return AnnotationSet.model_validate(
        {
            "events": [
                {
                    "event_id": "event-1",
                    "annotator": "annotator-a",
                    "mechanism": "knee_related_pattern",
                    "angles": [
                        {
                            "video_sha256": sha,
                            "onset_time_s": 0.45,
                            "contact_status": "direct",
                            "first_change_side": "left",
                        }
                    ],
                }
            ]
        }
    )


def test_scores_match_hand_computed_values(hand_run):
    angle_rows, run_rows = score_run(hand_run, _labels())
    observer, blind = angle_rows

    assert observer.stage == "observer"
    assert observer.onset_predicted_s == pytest.approx(0.5)
    assert observer.onset_error_s == pytest.approx(0.05)
    assert observer.onset_error_frames == pytest.approx(1.5)
    assert observer.contact_correct is True
    assert observer.side_correct is True

    assert blind.onset_error_s == pytest.approx(0.2)
    assert blind.onset_error_frames == pytest.approx(6.0)
    assert blind.contact_predicted == "unclear" and blind.contact_correct is False
    assert blind.side_correct is False

    (run,) = run_rows
    assert run.mechanism_correct is True
    assert run.judge_claims == 2
    assert run.judge_claims_citing_uncited_frames == 1
    assert (run.model_calls, run.input_tokens, run.output_tokens) == (4, 4000, 800)
    assert run.model_latency_s == pytest.approx(6.0)


def test_scoring_refuses_labels_for_a_different_video(hand_run):
    with pytest.raises(InjuryAnalysisError, match="No labelled event matches"):
        score_run(hand_run, _labels(sha="b" * 64))


def test_unclear_human_side_is_not_scored(hand_run):
    labels = _labels()
    labels.events[0].angles[0].first_change_side = BodySide.UNCLEAR
    angle_rows, _ = score_run(hand_run, labels)
    assert all(row.side_correct is None for row in angle_rows)


def test_pipeline_never_opens_the_labels(synthetic_video, tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "annotations.json").write_text('{"events": []}')
    monkeypatch.chdir(tmp_path)

    opened: list[str] = []
    state = {"active": True}

    def hook(event, args):
        if state["active"] and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
            opened.append(str(args[0]))

    sys.addaudithook(hook)
    try:
        config = RunConfig(
            video_path=synthetic_video,
            output_dir=tmp_path / "run",
            broad_fps=4,
            focused_fps=12,
            mock=True,
            max_individual_images=40,
        )
        run_pipeline(config, adapter=MockModelAdapter())
    finally:
        state["active"] = False

    assert opened, "the audit hook saw no file opens"
    assert not [path for path in opened if "eval" in Path(path).parts or path.endswith("annotations.json")]


def test_score_cli_end_to_end_on_a_mock_run(synthetic_video, tmp_path, capsys):
    run = tmp_path / "mock-run"
    config = RunConfig(
        video_path=synthetic_video,
        output_dir=run,
        broad_fps=4,
        focused_fps=12,
        mock=True,
        max_individual_images=40,
    )
    run_pipeline(config, adapter=MockModelAdapter())

    assert main(["fingerprint", "--video", str(synthetic_video)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(file_sha256(synthetic_video))

    labels = tmp_path / "annotations.json"
    labels.write_text(_labels(sha=file_sha256(synthetic_video)).model_dump_json())
    output = tmp_path / "scores.json"
    assert main(["score", "--annotations", str(labels), "--runs", str(run), "--output", str(output)]) == 0
    summary = capsys.readouterr().out
    assert "| mock-run | observer |" in summary
    scores = json.loads(output.read_text())
    assert {row["stage"] for row in scores["angles"]} == {"observer", "blind_reanalysis"}
    assert scores["runs"][0]["model_calls"] == 4

    assert main(["score", "--annotations", str(labels), "--runs", str(tmp_path / "missing")]) == 2


def test_example_labels_file_is_valid():
    example = Path(__file__).resolve().parents[1] / "eval" / "annotations.example.json"
    labels = AnnotationSet.model_validate_json(example.read_text())
    assert labels.events and labels.events[0].angles
