from __future__ import annotations

import re

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.schemas import (
    Candidate,
    CandidateReport,
    Confidence,
    JudgeReport,
    MechanismCategory,
    ObserverReport,
)
from injury_analysis.video.frames import FrameManifest

_DIAGNOSTIC_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btore\b",
        r"\btorn\b",
        r"\bruptures?\b",
        r"\bruptured\b",
        r"\bgrades?\s*[0-9]+\b",
        r"\bdiagnos(?:is|ed|tic|e)\b",
        r"\bconfirmed\b",
        r"\btears?\b",
        r"\bstrains?\b",
        r"\bseverity\b",
        r"\bsurgery\b",
        r"\bMRI\b",
        r"\bthis confirms\b",
        r"\btissue damage\b",
        r"\brecovery time\b",
        r"\bdefinitely\b",
    )
]


def validate_observer(report: ObserverReport, manifest: FrameManifest) -> list[str]:
    errors: list[str] = []
    ids = [item.observation_id for item in report.observations]
    if len(ids) != len(set(ids)):
        errors.append("observation_id values must be unique")
    known = set(manifest.by_id())
    for observation in report.observations:
        unknown = [frame_id for frame_id in observation.supporting_frame_ids if frame_id not in known]
        if unknown:
            errors.append(
                f"{observation.observation_id} cites unknown frame IDs: {', '.join(unknown)}"
            )
        errors.extend(_one_angle(observation.observation_id, observation.supporting_frame_ids, manifest))
        errors.extend(_lint(observation.observation_id, _strings(observation)))
    errors.extend(_lint("summary", [report.summary, *report.global_visibility_limitations]))
    return errors


INSUFFICIENT_CANDIDATE_ID = "c-insufficient"


def normalize_candidate(report: CandidateReport, observer: ObserverReport) -> CandidateReport:
    """Enforce "exactly one insufficient_visual_evidence candidate" in code instead of a retry.

    The rule is structural, so a second model call cannot do better than code: a missing
    candidate is added (ranked last), and extra ones are dropped after the first. Every
    change is written to the report's notes so the report shows what code did.
    """
    insufficient = [item for item in report.candidates if item.category == MechanismCategory.INSUFFICIENT]
    if len(insufficient) == 1:
        return report
    notes = list(report.notes)
    if not insufficient:
        frame_ids = sorted({fid for item in observer.observations for fid in item.supporting_frame_ids})
        taken = {item.candidate_id for item in report.candidates}
        candidate_id = INSUFFICIENT_CANDIDATE_ID
        suffix = 2
        while candidate_id in taken:
            candidate_id = f"{INSUFFICIENT_CANDIDATE_ID}-{suffix}"
            suffix += 1
        added = Candidate(
            candidate_id=candidate_id,
            name="Insufficient visual evidence",
            category=MechanismCategory.INSUFFICIENT,
            mechanism_description=(
                "The cited frames may not show enough to tell the ranked visible-mechanism patterns apart."
            ),
            distinguishing_evidence_needed=["Closer or additional camera angles around the onset frames"],
            confidence=Confidence.LOW,
            frame_ids=frame_ids,
        )
        notes.append(
            f"{candidate_id} was added by code: the model returned no insufficient_visual_evidence candidate."
        )
        return report.model_copy(update={"candidates": [*report.candidates, added], "notes": notes})
    keep = insufficient[0]
    dropped = [item.candidate_id for item in insufficient[1:]]
    candidates = [
        item
        for item in report.candidates
        if item.category != MechanismCategory.INSUFFICIENT or item is keep
    ]
    notes.append(
        f"Code kept {keep.candidate_id} as the only insufficient_visual_evidence candidate "
        f"and dropped {', '.join(dropped)}."
    )
    return report.model_copy(update={"candidates": candidates, "notes": notes})


def validate_candidate(
    report: CandidateReport,
    manifest: FrameManifest,
    observer: ObserverReport,
) -> list[str]:
    errors: list[str] = []
    known_frames = set(manifest.by_id())
    known_observations = {item.observation_id for item in observer.observations}
    insufficient = [
        item for item in report.candidates if item.category == MechanismCategory.INSUFFICIENT
    ]
    if len(insufficient) != 1:
        errors.append("Include exactly one candidate with category insufficient_visual_evidence")
    candidate_ids = [item.candidate_id for item in report.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate_id values must be unique")
    for candidate in report.candidates:
        if not candidate.frame_ids:
            errors.append(f"{candidate.candidate_id} must cite frame IDs")
        if candidate.category != MechanismCategory.INSUFFICIENT and not candidate.observation_ids:
            errors.append(f"{candidate.candidate_id} must cite observation IDs")
        for observation_id in candidate.observation_ids:
            if observation_id not in known_observations:
                errors.append(f"{candidate.candidate_id} cites unknown observation ID {observation_id}")
        unknown_frames = [frame_id for frame_id in candidate.frame_ids if frame_id not in known_frames]
        if unknown_frames:
            errors.append(
                f"{candidate.candidate_id} cites unknown frame IDs: {', '.join(unknown_frames)}"
            )
        for link in [*candidate.supporting_evidence, *candidate.weakening_evidence]:
            if link.observation_id not in known_observations:
                errors.append(
                    f"{candidate.candidate_id} evidence cites unknown observation ID {link.observation_id}"
                )
            missing = [frame_id for frame_id in link.frame_ids if frame_id not in known_frames]
            if missing:
                errors.append(
                    f"{candidate.candidate_id} evidence cites unknown frame IDs: {', '.join(missing)}"
                )
        errors.extend(_lint(candidate.candidate_id, _strings(candidate)))
    errors.extend(_lint("summary", [report.summary, *report.notes]))
    return errors


def validate_judge(report: JudgeReport, manifest: FrameManifest) -> list[str]:
    errors: list[str] = []
    known = set(manifest.by_id())
    claim_ids = [item.claim_id for item in report.claim_assessments]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("claim_id values must be unique")
    for claim in report.claim_assessments:
        unknown = [frame_id for frame_id in claim.frame_ids if frame_id not in known]
        if unknown:
            errors.append(f"{claim.claim_id} cites unknown frame IDs: {', '.join(unknown)}")
        errors.extend(_one_angle(claim.claim_id, claim.frame_ids, manifest))
        errors.extend(_lint(claim.claim_id, _strings(claim)))
    errors.extend(
        _lint(
            "judge",
            [
                *report.timestamp_disagreements,
                *report.unsupported_claims,
                *report.missing_plausible_alternatives,
                report.ranking_rationale,
                *report.limitations,
            ],
        )
    )
    return errors


def _one_angle(label: str, frame_ids: list[str], manifest: FrameManifest) -> list[str]:
    angles = {
        manifest.frame(frame_id).clip_id
        for frame_id in frame_ids
        if frame_id in manifest.by_id() and manifest.frame(frame_id).clip_id
    }
    if len(angles) > 1:
        return [
            f"{label} cites frames from more than one angle ({', '.join(sorted(angles))}). "
            "Cite one angle per item. Each angle has its own clock."
        ]
    return []


def authoritative_span(
    proposed_start: float,
    proposed_end: float,
    frame_ids: list[str],
    manifest: FrameManifest,
) -> tuple[float, float, str | None]:
    frames = [manifest.frame(frame_id) for frame_id in frame_ids]
    angles = {frame.clip_id for frame in frames if frame.clip_id}
    if len(angles) > 1:
        # Validation rejects these before binding. Never fall back to model-generated times.
        raise InjuryAnalysisError(
            "Cannot bind one time span to frames from "
            + ", ".join(sorted(angles))
            + ". Those angles do not share a clock."
        )
    times = [frame.time_s for frame in frames]
    start = min(times)
    end = max(times)
    note = None
    if abs(proposed_start - start) > 0.05 or abs(proposed_end - end) > 0.05:
        cited = ", ".join(frame_ids)
        note = (
            f"model times {proposed_start:.3f}-{proposed_end:.3f} replaced with "
            f"presentation timestamps {start:.3f}-{end:.3f} from {cited}"
        )
    return start, end, note


def bind_observer(report: ObserverReport, manifest: FrameManifest) -> tuple[ObserverReport, list[str]]:
    notes: list[str] = []
    updated = []
    for observation in report.observations:
        start, end, note = authoritative_span(
            observation.start_time_s,
            observation.end_time_s,
            observation.supporting_frame_ids,
            manifest,
        )
        if note:
            notes.append(f"{observation.observation_id}: {note}")
        updated.append(observation.model_copy(update={"start_time_s": start, "end_time_s": end}))
    return report.model_copy(update={"observations": updated}), notes


def bind_judge(report: JudgeReport, manifest: FrameManifest) -> tuple[JudgeReport, list[str]]:
    notes: list[str] = []
    updated = []
    for claim in report.claim_assessments:
        start, end, note = authoritative_span(
            claim.start_time_s,
            claim.end_time_s,
            claim.frame_ids,
            manifest,
        )
        if note:
            notes.append(f"{claim.claim_id}: {note}")
        updated.append(claim.model_copy(update={"start_time_s": start, "end_time_s": end}))
    return report.model_copy(update={"claim_assessments": updated}), notes


# Terms the diagnostic lint looks for. Kept in one place so the negation rules below
# cover every term the lint can flag.
_DIAGNOSTIC_TERMS = (
    r"confirmed|confirms|severity|tears?|strains?|torn|tore|ruptures?|ruptured|"
    r"diagnos(?:is|ed|tic|e)|tissue\s+damage|grades?\s*[0-9]+|recovery\s+time|surgery|MRI|definitely"
)
_NEGATORS = (
    r"not|no|never|without|cannot|can't|cant|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|"
    r"neither|nor|none|nothing|unable|lacks?|lacking|insufficient|beyond"
)
# A negation reaches at most four words and never past a comma or a conjunction, so
# "no contact and the hamstring is torn" is still flagged.
_CLAUSE_WORD = r"\s+(?!(?:and|but|while|although|though|yet|so|because)\b)[\w/-]+"
# "cannot show tissue damage", "no visible tear", "without a confirmed ..."
_NEGATED_BEFORE = re.compile(
    rf"\b(?:{_NEGATORS})\b(?:{_CLAUSE_WORD}){{0,4}}?\s+\b(?:{_DIAGNOSTIC_TERMS})\b",
    re.IGNORECASE,
)
# "tissue damage cannot be determined", "severity is not visible", "a tear was not seen".
# Requires a verb before the negation so "torn, not strained" is still flagged.
_NEGATED_AFTER = re.compile(
    rf"\b(?:{_DIAGNOSTIC_TERMS})\b(?:\s+[\w/-]+){{0,3}}?\s+"
    r"(?:cannot|can't|could\s+not|is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|"
    r"wasn't|weren't|would\s+not|will\s+not|is\s+never|remains?\s+(?:unknown|unclear|undetermined))\b",
    re.IGNORECASE,
)
# "would require MRI", "needs imaging or a diagnosis to ..." describe what video cannot do.
_REQUIRES_OUTSIDE_EVIDENCE = re.compile(
    r"\b(?:requires?|required|needs?|needed|would\s+need)\b(?:\s+[\w/-]+){0,3}?\s+"
    r"\b(?:MRI|imaging|diagnos(?:is|tic|e)|clinical)\b",
    re.IGNORECASE,
)


def strip_negated_terms(text: str) -> str:
    """Remove phrases where a diagnostic term is denied or deferred, so the lint only sees assertions.

    Models that follow the instructions write cautious sentences such as "video cannot show
    tissue damage". Those must not trigger a retry; "the hamstring is torn" must.
    """
    cleaned = _REQUIRES_OUTSIDE_EVIDENCE.sub(" ", text)
    cleaned = _NEGATED_BEFORE.sub(" ", cleaned)
    cleaned = _NEGATED_AFTER.sub(" ", cleaned)
    return re.sub(r"\bunconfirmed\b", " ", cleaned, flags=re.IGNORECASE)


def _lint(label: str, texts: list[str]) -> list[str]:
    errors: list[str] = []
    for text in texts:
        cleaned = strip_negated_terms(text)
        for pattern in _DIAGNOSTIC_PATTERNS:
            if pattern.search(cleaned):
                errors.append(
                    f"{label} contains diagnostic language matching '{pattern.pattern}'. "
                    "Describe visible movement or a mechanism pattern only."
                )
                break
    return errors


def _strings(model) -> list[str]:
    return _collect(model.model_dump(mode="json"))


def _collect(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        collected: list[str] = []
        for item in value:
            collected.extend(_collect(item))
        return collected
    if isinstance(value, dict):
        collected = []
        for item in value.values():
            collected.extend(_collect(item))
        return collected
    return []
