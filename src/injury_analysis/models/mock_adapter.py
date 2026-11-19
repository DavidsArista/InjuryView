from __future__ import annotations

import json

from injury_analysis.constants import BLIND_BLOCK, CANDIDATE_BLOCK, OBSERVER_BLOCK
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.model_adapter import ModelRequest
from injury_analysis.models.schemas import (
    BodySide,
    Candidate,
    CandidateReport,
    ClaimAssessment,
    ClaimVerdict,
    Confidence,
    ContactStatus,
    EvidenceLink,
    JudgeReport,
    MechanismCategory,
    Observation,
    ObserverReport,
    Phase,
    RankingDecision,
    StatementKind,
)
from injury_analysis.stages.compare import compare_observations

FIRST_PASS_MARKER = "MOCK_FIRST_PASS_RIGHT_SIDE"
BLIND_PASS_MARKER = "MOCK_BLIND_PASS_LEFT_SIDE"
CANDIDATE_MARKER = "MOCK_CANDIDATE_MARKER"

_PHASES = [
    Phase.BEFORE,
    Phase.ONSET,
    Phase.DESCENT,
    Phase.IMMEDIATE_REACTION,
    Phase.POST_EVENT,
]


class MockModelAdapter:
    """Deterministic fixture model.

    Responses are schema-valid and cite real frame IDs, but they do not inspect
    pixels and must not be read as an analysis of the clip.
    """

    def __init__(self):
        self.calls: list[ModelRequest] = []

    def complete(self, request: ModelRequest):
        self.calls.append(request)
        if request.stage == "observer":
            return _observer(request, body_side=BodySide.RIGHT, marker=FIRST_PASS_MARKER, extra=False)
        if request.stage == "blind_reanalysis":
            return _observer(request, body_side=BodySide.LEFT, marker=BLIND_PASS_MARKER, extra=True)
        if request.stage == "candidate":
            return _candidate(request)
        if request.stage == "judge":
            return _judge(request)
        raise InjuryAnalysisError(f"Mock model has no fixture for stage {request.stage}")


def _observer(request: ModelRequest, *, body_side: BodySide, marker: str, extra: bool) -> ObserverReport:
    frames = list(request.manifest.frames)
    if not frames:
        raise InjuryAnalysisError("Mock observer requires extracted frames")
    grouped = _group_frames(frames)
    observations = []
    for index, (phase, group) in enumerate(grouped, start=1):
        observations.append(
            Observation(
                observation_id=f"o{index:03d}",
                start_time_s=0.0,
                end_time_s=0.0,
                phase=phase,
                player="unclear",
                body_region=f"{body_side.value} lower limb",
                body_side=body_side,
                visible_movement=(
                    f"{marker} Fixture response only. The mock model did not inspect pixels. "
                    f"This interval is assigned to phase {phase.value}. "
                    f"Body side in this fixture is {body_side.value}."
                ),
                contact_status=ContactStatus.UNCLEAR,
                supporting_frame_ids=[frame.frame_id for frame in group],
                confidence=Confidence.LOW,
                visibility_limitations=["Mock model does not inspect pixels."],
                statement_kind=StatementKind.INFERRED,
            )
        )
    if extra:
        observations.append(
            Observation(
                observation_id="o-extra-blind",
                start_time_s=0.0,
                end_time_s=0.0,
                phase=Phase.POST_EVENT,
                player="unclear",
                body_region=f"{body_side.value} lower limb",
                body_side=body_side,
                visible_movement=(
                    f"{marker} Extra fixture note present only in the blind pass. "
                    "The mock model did not inspect pixels."
                ),
                contact_status=ContactStatus.UNCLEAR,
                supporting_frame_ids=[frames[-1].frame_id],
                confidence=Confidence.LOW,
                visibility_limitations=["Mock model does not inspect pixels."],
                statement_kind=StatementKind.INFERRED,
            )
        )
    covered = {observation.phase for observation in observations}
    return ObserverReport(
        summary=f"{marker} Fixture observer summary. This is not a reading of the video.",
        observations=observations,
        phases_not_visible=[phase for phase in _PHASES if phase not in covered],
        global_visibility_limitations=["Mock model does not inspect pixels."],
    )


def _candidate(request: ModelRequest) -> CandidateReport:
    payload = _extract_block(request.user_text, OBSERVER_BLOCK)
    observer = ObserverReport.model_validate(payload)
    first = observer.observations[0]
    frame_ids = list(first.supporting_frame_ids[:1])
    link = EvidenceLink(
        observation_id=first.observation_id,
        frame_ids=frame_ids,
        note="The mock analyzer can cite the fixture timeline only.",
    )
    insufficient = Candidate(
        candidate_id="k001",
        name="Insufficient visual evidence",
        category=MechanismCategory.INSUFFICIENT,
        mechanism_description=(
            f"{CANDIDATE_MARKER} The fixture timeline does not identify a visible mechanism."
        ),
        supporting_evidence=[],
        weakening_evidence=[link],
        distinguishing_evidence_needed=["A vision-model description of the actual frames."],
        confidence=Confidence.MODERATE,
        observation_ids=[first.observation_id],
        frame_ids=frame_ids,
        unsupported_assumptions=["No pixel evidence was reviewed by this fixture."],
    )
    unspecified = Candidate(
        candidate_id="k002",
        name="Unspecified movement pattern",
        category=MechanismCategory.OTHER,
        mechanism_description=(
            "An unspecified movement pattern remains possible because the fixture did not view the frames."
        ),
        supporting_evidence=[link],
        weakening_evidence=[
            EvidenceLink(
                observation_id=first.observation_id,
                frame_ids=frame_ids,
                note="Nothing in the fixture timeline distinguishes one mechanism from another.",
            )
        ],
        distinguishing_evidence_needed=[
            "Clearer views of the limb, any contact, and the moment movement changes."
        ],
        confidence=Confidence.LOW,
        observation_ids=[first.observation_id],
        frame_ids=frame_ids,
        unsupported_assumptions=["Treating a fixture timeline as movement evidence would be unsupported."],
    )
    return CandidateReport(
        summary=(
            f"{CANDIDATE_MARKER} Fixture candidate ranking. "
            "The mock model does not interpret video pixels."
        ),
        candidates=[insufficient, unspecified],
        notes=["Leading conclusion is insufficient visual evidence because no pixels were reviewed."],
    )


def _judge(request: ModelRequest) -> JudgeReport:
    observer = ObserverReport.model_validate(_extract_block(request.user_text, OBSERVER_BLOCK))
    candidate = CandidateReport.model_validate(_extract_block(request.user_text, CANDIDATE_BLOCK))
    blind = ObserverReport.model_validate(_extract_block(request.user_text, BLIND_BLOCK))
    comparison = compare_observations(observer, blind, request.manifest)
    assessments = []
    for index, item in enumerate(candidate.candidates, start=1):
        frame_ids = list(item.frame_ids) or list(observer.observations[0].supporting_frame_ids[:1])
        verdict = (
            ClaimVerdict.PARTIALLY_SUPPORTED
            if item.category == MechanismCategory.INSUFFICIENT
            else ClaimVerdict.NOT_VISIBLE
        )
        assessments.append(
            ClaimAssessment(
                claim_id=f"c{index:03d}",
                claim_text=item.mechanism_description,
                verdict=verdict,
                rationale=(
                    "Fixture judge: the mock model does not inspect pixels, so mechanism "
                    "language is not treated as something shown by the frames."
                ),
                frame_ids=frame_ids,
                start_time_s=0.0,
                end_time_s=0.0,
                related_observation_ids=list(item.observation_ids),
            )
        )
    return JudgeReport(
        claim_assessments=assessments,
        reproduced_by_both_passes=["Both fixture passes cover the extracted frame span."],
        only_in_first_pass=comparison.only_in_first,
        only_in_second_pass=comparison.only_in_second,
        timestamp_disagreements=[],
        body_side_disagreements=comparison.body_side_disagreements,
        unsupported_claims=["Fixture pattern language is not a pixel observation."],
        missing_plausible_alternatives=["A vision-model pass over the extracted frames is still required."],
        ranking_decision=RankingDecision.INCONCLUSIVE,
        ranking_rationale=(
            "The fixture passes disagree on body side and neither pass inspected the pixels, "
            "so the ranking stays inconclusive."
        ),
        limitations=["Mock judge compared structured fixture fields and did not view the images."],
    )


def _group_frames(frames):
    """Phase groups within each angle, so no fixture observation mixes angle clocks."""
    by_angle: dict[str, list] = {}
    for frame in frames:
        by_angle.setdefault(frame.clip_id, []).append(frame)
    groups = []
    for angle_frames in by_angle.values():
        groups.extend(_group_angle_frames(angle_frames))
    return groups


def _group_angle_frames(frames):
    if len(frames) < len(_PHASES):
        return [(_PHASES[index], [frame]) for index, frame in enumerate(frames)]
    groups: list[list] = [[] for _ in _PHASES]
    for index, frame in enumerate(frames):
        bucket = min(len(_PHASES) - 1, index * len(_PHASES) // len(frames))
        groups[bucket].append(frame)
    return [(phase, group) for phase, group in zip(_PHASES, groups, strict=True) if group]


def _extract_block(text: str, tag: str) -> dict:
    start = f"<{tag}>"
    end = f"</{tag}>"
    try:
        opening = text.index(start)
        closing = text.index(end, opening + len(start))
    except ValueError as exc:
        raise InjuryAnalysisError(f"Mock model could not find <{tag}> in the prompt") from exc
    payload = text[opening + len(start) : closing].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InjuryAnalysisError(f"Mock model found <{tag}> but the JSON was invalid") from exc
    if not isinstance(parsed, dict):
        raise InjuryAnalysisError(f"<{tag}> did not contain a JSON object")
    return parsed
