import os

import json

import pytest
from pydantic import ValidationError

from injury_analysis.config import load_env_file
from injury_analysis.models.anthropic_adapter import inline_schema
from injury_analysis.models.schemas import (
    BodySide,
    Candidate,
    CandidateReport,
    Confidence,
    ContactStatus,
    EvidenceLink,
    MechanismCategory,
    Observation,
    ObserverReport,
    Phase,
    StatementKind,
)
from injury_analysis.models.validation import bind_observer, validate_candidate, validate_observer
from injury_analysis.stages.prompts import CANDIDATE_SYSTEM, JUDGE_SYSTEM, OBSERVER_SYSTEM
from injury_analysis.video.frames import FrameManifest, FrameRecord


def _manifest() -> FrameManifest:
    return FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                path="frames/f001.jpg",
                width=8,
                height=8,
                sampling="broad",
            ),
            FrameRecord(
                frame_id="f002",
                source_frame_index=8,
                pts_s=0.25,
                time_s=0.25,
                path="frames/f002.jpg",
                width=8,
                height=8,
                sampling="broad",
            ),
        ],
    )


def _observation(**overrides) -> Observation:
    payload = dict(
        observation_id="o001",
        start_time_s=999.0,
        end_time_s=1000.0,
        phase=Phase.ONSET,
        player="unclear",
        body_region="lower limb",
        body_side=BodySide.UNCLEAR,
        visible_movement="The stride changes between the cited frames.",
        contact_status=ContactStatus.UNCLEAR,
        supporting_frame_ids=["f001", "f002"],
        confidence=Confidence.LOW,
        visibility_limitations=["motion blur"],
        statement_kind=StatementKind.OBSERVED,
    )
    payload.update(overrides)
    return Observation(**payload)


def _report(*observations: Observation) -> ObserverReport:
    return ObserverReport(summary="Visible movement only.", observations=list(observations))


def test_schema_accepts_an_observation_and_rejects_extra_fields():
    report = _report(_observation())
    restored = ObserverReport.model_validate_json(report.model_dump_json())
    assert restored.observations[0].statement_kind == StatementKind.OBSERVED

    payload = report.model_dump()
    payload["diagnosis"] = "not allowed"
    with pytest.raises(ValidationError):
        ObserverReport.model_validate(payload)


def test_confidence_is_categorical():
    with pytest.raises(ValidationError):
        _observation(confidence="95%")


def test_denying_confirmation_is_allowed_and_asserting_it_is_not():
    manifest = _manifest()
    denied = validate_observer(
        _report(_observation(visible_movement="The frames do not show a confirmed injury.")),
        manifest,
    )
    asserted = validate_observer(
        _report(_observation(visible_movement="The injury is confirmed.")),
        manifest,
    )
    assert denied == []
    assert asserted


def test_observer_validation_rejects_unknown_frames_and_diagnostic_language():
    manifest = _manifest()
    assert validate_observer(_report(_observation()), manifest) == []

    unknown = validate_observer(
        _report(_observation(supporting_frame_ids=["not-a-frame"])),
        manifest,
    )
    assert any("unknown frame" in error for error in unknown)

    diagnostic = validate_observer(
        _report(_observation(visible_movement="He tore his Achilles and it is grade 2.")),
        manifest,
    )
    assert diagnostic


def test_binding_replaces_invented_timestamps_with_frame_times():
    manifest = _manifest()
    bound, notes = bind_observer(_report(_observation()), manifest)

    assert bound.observations[0].start_time_s == pytest.approx(0.0)
    assert bound.observations[0].end_time_s == pytest.approx(0.25)
    assert notes
    assert "999.000" in notes[0]


def test_candidate_report_must_include_insufficient_evidence_once():
    manifest = _manifest()
    observer = _report(_observation())
    link = EvidenceLink(observation_id="o001", frame_ids=["f001"], note="The frames show a change.")
    insufficient = Candidate(
        candidate_id="k001",
        name="Insufficient visual evidence",
        category=MechanismCategory.INSUFFICIENT,
        mechanism_description="The frames do not distinguish a mechanism.",
        weakening_evidence=[link],
        distinguishing_evidence_needed=["Another angle."],
        confidence=Confidence.MODERATE,
        observation_ids=["o001"],
        frame_ids=["f001"],
        unsupported_assumptions=["None beyond the sampled frames."],
    )
    other = Candidate(
        candidate_id="k002",
        name="Other visible pattern",
        category=MechanismCategory.OTHER,
        mechanism_description="Some other visible pattern is still possible.",
        supporting_evidence=[link],
        distinguishing_evidence_needed=["A clearer view of the limb."],
        confidence=Confidence.LOW,
        observation_ids=["o001"],
        frame_ids=["f002"],
        unsupported_assumptions=["The pattern name is only a placeholder."],
    )
    valid = CandidateReport(summary="Two possibilities.", candidates=[insufficient, other])
    assert validate_candidate(valid, manifest, observer) == []

    missing = CandidateReport(summary="Missing the required conclusion.", candidates=[other])
    assert any("insufficient_visual_evidence" in error for error in validate_candidate(missing, manifest, observer))


def test_mechanism_taxonomy_is_in_the_schema_not_steered_by_the_prompt():
    values = {item.value for item in MechanismCategory}
    assert "insufficient_visual_evidence" in values
    assert "posterior_thigh_hamstring_pattern" in values
    assert "calf_achilles_complex_pattern" in values

    prose = "\n".join([OBSERVER_SYSTEM, CANDIDATE_SYSTEM, JUDGE_SYSTEM]).lower()
    for phrase in (
        "caleb",
        "williams",
        "youtube",
        "hamstring",
        "achilles",
        "-ysudwbehxk",
        "publicly reported",
    ):
        assert phrase not in prose
    assert "do not try to make the two runs agree" in JUDGE_SYSTEM.lower()


def test_env_file_fills_a_missing_key_without_overriding(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text('# comment\nexport ANTHROPIC_API_KEY="test-key"\nANTHROPIC_API_KEY_EMPTY=\n')
    load_env_file(env_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")
    env_path.write_text("ANTHROPIC_API_KEY=other-key\n")
    load_env_file(env_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "already-set"


def test_tool_schema_inlines_local_references():
    schema = inline_schema(ObserverReport.model_json_schema())
    assert "$ref" not in json.dumps(schema)
    assert "observations" in schema["properties"]
