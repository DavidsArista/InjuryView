"""Checks for the two causes of avoidable retries seen in live runs.

1. The diagnostic-language lint flagged cautious, negated sentences
   ("cannot show tissue damage", "cannot be confirmed").
2. The "exactly one insufficient_visual_evidence candidate" rule was enforced by a retry,
   although code can apply it without asking the model again.
"""

from __future__ import annotations

import pytest

from injury_analysis.models.model_adapter import ModelRequest, complete_with_retry
from injury_analysis.models.schemas import (
    Candidate,
    CandidateReport,
    Confidence,
    MechanismCategory,
    ObserverReport,
)
from injury_analysis.models.validation import (
    INSUFFICIENT_CANDIDATE_ID,
    _lint,
    normalize_candidate,
    validate_candidate,
)
from injury_analysis.video.frames import FrameManifest, FrameRecord


@pytest.mark.parametrize(
    "text",
    [
        "Video alone cannot show tissue damage.",
        "Tissue damage cannot be determined from these frames.",
        "There is no visible tear.",
        "Severity is not visible in any frame.",
        "Distinguishing these patterns would require MRI.",
        "The report does not present the injury as confirmed.",
        "The claim is unconfirmed.",
        "Insufficient frames to show a strain pattern.",
        "Without a confirmed contact frame, the onset stays uncertain.",
    ],
)
def test_negated_or_deferred_terms_do_not_trigger_the_lint(text):
    assert _lint("x", [text]) == []


@pytest.mark.parametrize(
    "text",
    [
        "The hamstring is torn, not strained.",
        "No contact and the hamstring is torn.",
        "This confirms a hamstring strain.",
        "Grade 2 strain of the left calf.",
        "Expected recovery time is three weeks.",
        "MRI shows a rupture.",
        "The player definitely tore the ligament.",
        "A tear is visible at f031.",
    ],
)
def test_asserted_diagnostic_terms_are_still_flagged(text):
    assert _lint("x", [text])


def _manifest() -> FrameManifest:
    return FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id=f"f00{index}",
                source_frame_index=index,
                pts_s=index / 4,
                time_s=index / 4,
                path=f"frames/f00{index}.jpg",
                width=8,
                height=8,
                sampling="broad",
            )
            for index in (1, 2, 3)
        ],
    )


def _observer() -> ObserverReport:
    return ObserverReport.model_validate(
        {
            "summary": "Visible movement only.",
            "observations": [
                {
                    "observation_id": "o001",
                    "start_time_s": 0.25,
                    "end_time_s": 0.5,
                    "phase": "onset",
                    "player": "unclear",
                    "body_region": "lower limb",
                    "body_side": "left",
                    "visible_movement": "Stride shortens.",
                    "contact_status": "no_visible_contact",
                    "supporting_frame_ids": ["f001", "f002"],
                    "confidence": "low",
                    "statement_kind": "observed",
                }
            ],
        }
    )


def _candidate(candidate_id: str, category: MechanismCategory) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        name=candidate_id,
        category=category,
        mechanism_description="A visible-mechanism hypothesis.",
        distinguishing_evidence_needed=["A closer angle"],
        confidence=Confidence.LOW,
        observation_ids=["o001"],
        frame_ids=["f001"],
    )


def test_missing_insufficient_candidate_is_added_last_with_a_note():
    report = CandidateReport(
        summary="One pattern.",
        candidates=[_candidate("c1", MechanismCategory.KNEE)],
    )
    fixed = normalize_candidate(report, _observer())

    assert [item.candidate_id for item in fixed.candidates] == ["c1", INSUFFICIENT_CANDIDATE_ID]
    added = fixed.candidates[-1]
    assert added.category == MechanismCategory.INSUFFICIENT
    assert added.frame_ids == ["f001", "f002"]
    assert any("added by code" in note for note in fixed.notes)
    assert validate_candidate(fixed, _manifest(), _observer()) == []


def test_added_candidate_id_does_not_collide():
    report = CandidateReport(
        summary="One pattern.",
        candidates=[_candidate(INSUFFICIENT_CANDIDATE_ID, MechanismCategory.KNEE)],
    )
    fixed = normalize_candidate(report, _observer())
    assert fixed.candidates[-1].candidate_id == f"{INSUFFICIENT_CANDIDATE_ID}-2"


def test_extra_insufficient_candidates_are_dropped_after_the_first():
    report = CandidateReport(
        summary="Two insufficient entries.",
        candidates=[
            _candidate("c1", MechanismCategory.INSUFFICIENT),
            _candidate("c2", MechanismCategory.KNEE),
            _candidate("c3", MechanismCategory.INSUFFICIENT),
        ],
    )
    fixed = normalize_candidate(report, _observer())

    assert [item.candidate_id for item in fixed.candidates] == ["c1", "c2"]
    assert any("dropped c3" in note for note in fixed.notes)


def test_normalized_rule_costs_no_second_model_call():
    class OneShot:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return CandidateReport(summary="One pattern.", candidates=[_candidate("c1", MechanismCategory.KNEE)])

    adapter = OneShot()
    request = ModelRequest(
        stage="candidate",
        system="s",
        user_text="u",
        images=[],
        response_model=CandidateReport,
        manifest=_manifest(),
    )
    report, attempts = complete_with_retry(
        adapter,
        request,
        lambda item: validate_candidate(item, _manifest(), _observer()),
        normalize=lambda item: normalize_candidate(item, _observer()),
    )

    assert adapter.calls == 1
    assert len(attempts) == 1
    assert any(item.category == MechanismCategory.INSUFFICIENT for item in report.candidates)
