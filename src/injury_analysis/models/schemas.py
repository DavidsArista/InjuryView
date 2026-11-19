from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Phase(str, Enum):
    BEFORE = "before_anomaly"
    ONSET = "onset"
    DESCENT = "descent_or_loss_of_movement"
    IMMEDIATE_REACTION = "immediate_reaction"
    POST_EVENT = "post_event"


class ContactStatus(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    NO_VISIBLE_CONTACT = "no_visible_contact"
    UNCLEAR = "unclear"


class StatementKind(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


class Confidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class BodySide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDLINE = "midline"
    BILATERAL = "bilateral"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class MechanismCategory(str, Enum):
    POSTERIOR_THIGH_HAMSTRING = "posterior_thigh_hamstring_pattern"
    CALF_ACHILLES = "calf_achilles_complex_pattern"
    KNEE = "knee_related_pattern"
    ANKLE_FOOT = "ankle_foot_pattern"
    CRAMP = "cramp_or_transient_muscular_event"
    INSUFFICIENT = "insufficient_visual_evidence"
    OTHER = "other_visible_pattern"


class ClaimVerdict(str, Enum):
    DIRECTLY_SUPPORTED = "directly_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    NOT_VISIBLE = "not_visible"
    INTERPRETATION_AS_OBSERVATION = "interpretation_presented_as_observation"


class RankingDecision(str, Enum):
    RETAIN = "retain"
    CHANGE = "change"
    INCONCLUSIVE = "inconclusive"


class Observation(APIModel):
    observation_id: str = Field(min_length=1, description="Unique ID within this report, such as o001.")
    start_time_s: float = Field(description="Clip-relative seconds. Use only a timestamp printed for a cited frame.")
    end_time_s: float = Field(description="Clip-relative seconds. Use only a timestamp printed for a cited frame.")
    phase: Phase
    player: str = Field(min_length=1, description="Who is visible, or 'unclear'.")
    body_region: str = Field(min_length=1, description="Plain-language body region. No diagnosis.")
    body_side: BodySide
    visible_movement: str = Field(min_length=1, description="Literal visible movement. No diagnosis, grade, or tissue claim.")
    contact_status: ContactStatus
    supporting_frame_ids: list[str] = Field(min_length=1)
    confidence: Confidence
    visibility_limitations: list[str] = Field(default_factory=list)
    statement_kind: StatementKind


class ObserverReport(APIModel):
    summary: str = Field(min_length=1)
    observations: list[Observation] = Field(min_length=1)
    phases_not_visible: list[Phase] = Field(default_factory=list)
    global_visibility_limitations: list[str] = Field(default_factory=list)


class EvidenceLink(APIModel):
    observation_id: str = Field(min_length=1)
    frame_ids: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)


class Candidate(APIModel):
    candidate_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: MechanismCategory
    mechanism_description: str = Field(
        min_length=1,
        description="A visible-mechanism hypothesis. Do not claim confirmed tissue damage, grade, or recovery time.",
    )
    supporting_evidence: list[EvidenceLink] = Field(default_factory=list)
    weakening_evidence: list[EvidenceLink] = Field(default_factory=list)
    distinguishing_evidence_needed: list[str] = Field(min_length=1)
    confidence: Confidence
    observation_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    unsupported_assumptions: list[str] = Field(default_factory=list)


class CandidateReport(APIModel):
    summary: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1, description="Ordered from most plausible to least plausible.")
    notes: list[str] = Field(default_factory=list)


class ClaimAssessment(APIModel):
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    verdict: ClaimVerdict
    rationale: str = Field(min_length=1)
    frame_ids: list[str] = Field(min_length=1)
    start_time_s: float
    end_time_s: float
    related_observation_ids: list[str] = Field(default_factory=list)


class JudgeReport(APIModel):
    claim_assessments: list[ClaimAssessment] = Field(min_length=1)
    # Pairing the two passes (which observations match, which appear in one pass, body-side
    # disagreements) is computed by stages/compare.py and given to the judge as input.
    timestamp_disagreements: list[str] = Field(
        default_factory=list,
        description="Paired observations whose onset or span differs in a way that affects a claim.",
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_plausible_alternatives: list[str] = Field(default_factory=list)
    ranking_decision: RankingDecision
    ranking_rationale: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


Producer = Literal["mock", "anthropic", "openai", "muse"]


class ObserverArtifact(ObserverReport):
    producer: Producer
    model_name: str
    timestamp_authority: Literal["video_presentation_timestamp"]
    input_fingerprint: str | None = Field(
        default=None,
        description="Hash of the request that produced this artifact. Resume reuses an artifact only on an exact match.",
    )
    timestamp_adjustments: list[str] = Field(default_factory=list)


class CandidateArtifact(CandidateReport):
    producer: Producer
    model_name: str
    timestamp_authority: Literal["video_presentation_timestamp"]
    input_fingerprint: str | None = Field(
        default=None,
        description="Hash of the request that produced this artifact. Resume reuses an artifact only on an exact match.",
    )


# Artifacts written before 2026-11-19 also hold reproduced_by_both_passes, only_in_first_pass,
# only_in_second_pass and body_side_disagreements. Ignore those when reading old runs; the
# model-facing JudgeReport still forbids them.
_LEGACY_JUDGE_FIELDS = ConfigDict(extra="ignore", str_strip_whitespace=True)


class JudgeArtifact(JudgeReport):
    model_config = _LEGACY_JUDGE_FIELDS

    producer: Producer
    model_name: str
    timestamp_authority: Literal["video_presentation_timestamp"]
    input_fingerprint: str | None = Field(
        default=None,
        description="Hash of the request that produced this artifact. Resume reuses an artifact only on an exact match.",
    )
    timestamp_adjustments: list[str] = Field(default_factory=list)
