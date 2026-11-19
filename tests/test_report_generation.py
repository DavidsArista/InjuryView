from injury_analysis.constants import DISCLAIMER
from injury_analysis.models.schemas import (
    BodySide,
    Candidate,
    CandidateArtifact,
    ClaimAssessment,
    ClaimVerdict,
    Confidence,
    ContactStatus,
    EvidenceLink,
    JudgeArtifact,
    MechanismCategory,
    Observation,
    ObserverArtifact,
    Phase,
    RankingDecision,
    StatementKind,
)
from injury_analysis.stages.report import render_report
from injury_analysis.video.frames import FrameManifest, FrameRecord
from injury_analysis.video.metadata import VideoMetadata


def _artifacts():
    manifest = FrameManifest(
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.0,
        focus_end_s=0.5,
        source_offset_s=10,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                source_time_s=10.0,
                path="frames/f001.jpg",
                width=320,
                height=240,
                sampling="both",
            ),
            FrameRecord(
                frame_id="f002",
                source_frame_index=8,
                pts_s=0.25,
                time_s=0.25,
                source_time_s=10.25,
                path="frames/f002.jpg",
                width=320,
                height=240,
                sampling="focused",
            ),
        ],
    )
    metadata = VideoMetadata(
        video_path="/tmp/clip.mp4",
        duration_s=2.0,
        width=320,
        height=240,
        frame_rate=30.0,
        frame_rate_raw="30/1",
        time_base="1/15360",
        codec_name="h264",
        source_offset_s=10,
        probed_frame_count=60,
    )
    observer = ObserverArtifact(
        producer="mock",
        model_name="mock-fixture",
        timestamp_authority="video_presentation_timestamp",
        summary="Fixture summary.",
        observations=[
            Observation(
                observation_id="o001",
                start_time_s=0.0,
                end_time_s=0.25,
                phase=Phase.ONSET,
                player="unclear",
                body_region="lower limb",
                body_side=BodySide.RIGHT,
                visible_movement="The stride shortens across the cited frames.",
                contact_status=ContactStatus.NO_VISIBLE_CONTACT,
                supporting_frame_ids=["f001", "f002"],
                confidence=Confidence.LOW,
                visibility_limitations=["single angle"],
                statement_kind=StatementKind.OBSERVED,
            )
        ],
    )
    link = EvidenceLink(
        observation_id="o001",
        frame_ids=["f001"],
        note="The cited frames show a change and no readable contact.",
    )
    candidate = CandidateArtifact(
        producer="mock",
        model_name="mock-fixture",
        timestamp_authority="video_presentation_timestamp",
        summary="Two fixture possibilities.",
        candidates=[
            Candidate(
                candidate_id="k001",
                name="Insufficient visual evidence",
                category=MechanismCategory.INSUFFICIENT,
                mechanism_description="The frames do not establish a mechanism.",
                weakening_evidence=[link],
                distinguishing_evidence_needed=["Another camera angle."],
                confidence=Confidence.MODERATE,
                observation_ids=["o001"],
                frame_ids=["f001", "f002"],
                unsupported_assumptions=["The sample may miss the relevant motion."],
            )
        ],
    )
    blind = observer.model_copy(deep=True)
    judge = JudgeArtifact(
        producer="mock",
        model_name="mock-fixture",
        timestamp_authority="video_presentation_timestamp",
        claim_assessments=[
            ClaimAssessment(
                claim_id="c001",
                claim_text="The frames do not establish a mechanism.",
                verdict=ClaimVerdict.PARTIALLY_SUPPORTED,
                rationale="Both passes describe the same frames and leave the mechanism unresolved.",
                frame_ids=["f001", "f002"],
                start_time_s=0.0,
                end_time_s=0.25,
                related_observation_ids=["o001"],
            )
        ],
        ranking_decision=RankingDecision.INCONCLUSIVE,
        ranking_rationale="The visible record does not separate the possibilities.",
        unsupported_claims=["No unsupported diagnostic claim was introduced."],
    )
    return metadata, manifest, observer, candidate, blind, judge


def test_report_has_required_sections_labels_and_citations():
    metadata, manifest, observer, candidate, blind, judge = _artifacts()
    report = render_report(metadata, manifest, observer, candidate, blind, judge)

    for header in (
        "## 1. Video metadata",
        "## 2. Timestamped movement timeline",
        "## 3. Contact and movement observations",
        "## 4. Possible injury mechanisms",
        "## 5. Evidence supporting and weakening each possibility",
        "## 6. Independent-review findings",
        "## 7. Unsupported or disputed claims",
        "## 8. Overall conclusion",
        "## 9. Important limitations",
    ):
        assert header in report
    assert DISCLAIMER in report
    assert "OBSERVED" in report
    assert "INFERRED" in report
    assert "EXTERNAL GROUND TRUTH: none" in report
    assert "f001" in report
    assert "f002" in report
    assert "0.00s (source 10.00s)" in report
    assert "cannot confirm a specific injury" in report
    assert "mock fixture" in report.lower()
    assert "hamstring" not in report.lower()
    assert "achilles" not in report.lower()
