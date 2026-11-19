"""Score saved run artifacts against human labels.

This module only reads finished run directories and a labels file. The analysis pipeline never
imports it and never opens the labels, so labels cannot leak into a model prompt.

Model agreement is not treated as correct here. Every accuracy number compares a model field
with a human label for the same video, matched by content hash.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import Field

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.schemas import (
    APIModel,
    BodySide,
    CandidateArtifact,
    ContactStatus,
    JudgeArtifact,
    MechanismCategory,
    Observation,
    ObserverArtifact,
    Phase,
)
from injury_analysis.video.frames import FrameManifest

_FRAME_REFERENCE = re.compile(r"\bf\d{3}\b")


class AngleLabel(APIModel):
    video_sha256: str = Field(min_length=64, max_length=64, description="From `python -m injury_analysis fingerprint`.")
    angle: str = Field(default="", description="The --label used for this video, or empty for a single-clip run.")
    playback_speed: float = Field(default=1.0, gt=0, description="1.0 for live speed. A slow-motion replay is below 1.")
    onset_time_s: float | None = Field(
        default=None,
        description="Clip-relative seconds of the first visible movement change. Null when it cannot be seen.",
    )
    contact_status: ContactStatus | None = None
    first_change_side: BodySide | None = None


class EventLabel(APIModel):
    event_id: str = Field(min_length=1)
    annotator: str = Field(min_length=1)
    subject: str | None = None
    mechanism: MechanismCategory | None = Field(
        default=None,
        description="Optional. Leave null unless the frames alone let the annotator choose a category.",
    )
    angles: list[AngleLabel] = Field(min_length=1)


class AnnotationSet(APIModel):
    events: list[EventLabel]


@dataclass
class AngleScore:
    run: str
    event_id: str
    annotator: str
    stage: str
    angle: str
    onset_outcome: str
    onset_label_s: float | None
    onset_predicted_s: float | None
    onset_error_s: float | None
    onset_error_frames: float | None
    contact_label: str | None
    contact_predicted: str | None
    contact_correct: bool | None
    side_label: str | None
    side_predicted: str | None
    side_correct: bool | None


@dataclass
class RunScore:
    run: str
    event_id: str
    annotator: str
    mechanism_label: str | None
    leading_candidate: str
    mechanism_correct: bool | None
    judge_ranking_decision: str
    judge_claims: int
    judge_claims_citing_uncited_frames: int
    model_calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    model_latency_s: float | None


@dataclass
class _RunClip:
    angle: str
    video_sha256: str
    frame_rate: float


def load_annotations(path: Path) -> AnnotationSet:
    try:
        return AnnotationSet.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        raise InjuryAnalysisError(f"Could not read labels from {path}: {exc}") from exc


def score_runs(run_dirs: list[Path], annotations: AnnotationSet) -> tuple[list[AngleScore], list[RunScore]]:
    angle_rows: list[AngleScore] = []
    run_rows: list[RunScore] = []
    for run_dir in run_dirs:
        angles, runs = score_run(run_dir, annotations)
        angle_rows.extend(angles)
        run_rows.extend(runs)
    return angle_rows, run_rows


_REQUIRED = (
    "frame_manifest.json",
    "observer_report.json",
    "blind_reanalysis.json",
    "candidate_report.json",
    "judge_report.json",
)


def score_run(run_dir: Path, annotations: AnnotationSet) -> tuple[list[AngleScore], list[RunScore]]:
    missing = [name for name in _REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise InjuryAnalysisError(f"{run_dir} is not a finished run. Missing: {', '.join(missing)}")
    manifest = FrameManifest.model_validate_json((run_dir / "frame_manifest.json").read_text())
    clips = _run_clips(run_dir, manifest)
    run_hashes = {clip.video_sha256 for clip in clips}
    events = [
        event
        for event in annotations.events
        if run_hashes <= {angle.video_sha256 for angle in event.angles}
    ]
    if not events:
        raise InjuryAnalysisError(
            f"No labelled event matches the videos in {run_dir}. Each clip must be listed by its sha256; "
            "get it with `python -m injury_analysis fingerprint --video PATH`."
        )
    subject = _run_subject(run_dir)
    passes = {
        "observer": ObserverArtifact.model_validate_json((run_dir / "observer_report.json").read_text()),
        "blind_reanalysis": ObserverArtifact.model_validate_json((run_dir / "blind_reanalysis.json").read_text()),
    }
    candidate = CandidateArtifact.model_validate_json((run_dir / "candidate_report.json").read_text())
    judge = JudgeArtifact.model_validate_json((run_dir / "judge_report.json").read_text())
    usage = _usage(run_dir)

    angle_rows: list[AngleScore] = []
    run_rows: list[RunScore] = []
    for event in events:
        labels = {angle.video_sha256: angle for angle in event.angles}
        for stage, report in passes.items():
            for clip in clips:
                label = labels[clip.video_sha256]
                onset = _onset(report.observations, clip.angle, manifest, subject)
                angle_rows.append(_angle_score(run_dir.name, event, stage, clip, label, onset))
        leading = candidate.candidates[0].category
        run_rows.append(
            RunScore(
                run=run_dir.name,
                event_id=event.event_id,
                annotator=event.annotator,
                mechanism_label=None if event.mechanism is None else event.mechanism.value,
                leading_candidate=leading.value,
                mechanism_correct=None if event.mechanism is None else leading == event.mechanism,
                judge_ranking_decision=judge.ranking_decision.value,
                judge_claims=len(judge.claim_assessments),
                judge_claims_citing_uncited_frames=sum(
                    1
                    for claim in judge.claim_assessments
                    if set(_FRAME_REFERENCE.findall(claim.rationale)) - set(claim.frame_ids)
                ),
                **usage,
            )
        )
    return angle_rows, run_rows


def summarize(angle_rows: list[AngleScore], run_rows: list[RunScore]) -> str:
    """Markdown summary per run and stage. Counts are shown so small samples are not overstated."""
    lines = [
        "| Run | Stage | Onset found | Mean onset error (s) | Contact correct | Side correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    keys = sorted({(row.run, row.stage) for row in angle_rows})
    for run, stage in keys:
        rows = [row for row in angle_rows if row.run == run and row.stage == stage]
        errors = [row.onset_error_s for row in rows if row.onset_error_s is not None]
        labelled_onsets = [row for row in rows if row.onset_label_s is not None]
        found = sum(1 for row in labelled_onsets if row.onset_predicted_s is not None)
        mean_error = f"{sum(errors) / len(errors):.3f}" if errors else "n/a"
        lines.append(
            f"| {run} | {stage} | {found}/{len(labelled_onsets)} | {mean_error} | "
            f"{_ratio(row.contact_correct for row in rows)} | {_ratio(row.side_correct for row in rows)} |"
        )
    lines.extend(
        [
            "",
            "| Run | Event | Annotator | Leading candidate | Matches label | Judge decision | "
            "Judge claims citing frames outside their list | Model calls | Input tokens | Output tokens | Model latency (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| {row.run} | {row.event_id} | {row.annotator} | {row.leading_candidate} | "
            f"{_value(row.mechanism_correct)} | {row.judge_ranking_decision} | "
            f"{row.judge_claims_citing_uncited_frames}/{row.judge_claims} | {_value(row.model_calls)} | "
            f"{_value(row.input_tokens)} | {_value(row.output_tokens)} | {_value(row.model_latency_s)} |"
        )
    return "\n".join(lines) + "\n"


def write_scores(path: Path, angle_rows: list[AngleScore], run_rows: list[RunScore]) -> None:
    payload = {
        "angles": [asdict(row) for row in angle_rows],
        "runs": [asdict(row) for row in run_rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_clips(run_dir: Path, manifest: FrameManifest) -> list[_RunClip]:
    if manifest.clips:
        clips = [_RunClip(clip.clip_id, clip.video_sha256 or "", clip.frame_rate) for clip in manifest.clips]
    else:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        clips = [_RunClip("", manifest.video_sha256 or "", float(metadata["frame_rate"]))]
    if any(not clip.video_sha256 for clip in clips):
        raise InjuryAnalysisError(
            f"{run_dir} was produced before video fingerprints were recorded. Re-run it to score it."
        )
    return clips


def _run_subject(run_dir: Path) -> str | None:
    path = run_dir / "run_config.json"
    if not path.is_file():
        return None
    subjects = json.loads(path.read_text()).get("subjects") or []
    return subjects[0] if subjects else None


def _onset(
    observations: list[Observation],
    angle: str,
    manifest: FrameManifest,
    subject: str | None,
) -> Observation | None:
    """Earliest onset observation on this angle about the followed player, or an unnamed player."""
    frames = manifest.by_id()
    wanted = None if subject is None else _normalize(subject)
    matches = [
        observation
        for observation in observations
        if observation.phase == Phase.ONSET
        and {frames[frame_id].clip_id for frame_id in observation.supporting_frame_ids if frame_id in frames} == {angle}
        and (wanted is None or _normalize(observation.player) in (wanted, "unclear"))
    ]
    return min(matches, key=lambda item: (item.start_time_s, item.observation_id), default=None)


def _angle_score(
    run: str,
    event: EventLabel,
    stage: str,
    clip: _RunClip,
    label: AngleLabel,
    onset: Observation | None,
) -> AngleScore:
    predicted = None if onset is None else onset.start_time_s
    if label.onset_time_s is None:
        outcome = "both_absent" if predicted is None else "predicted_without_label"
        error = None
    elif predicted is None:
        outcome = "missed"
        error = None
    else:
        outcome = "both_present"
        error = abs(predicted - label.onset_time_s)
    contact = None if onset is None else onset.contact_status
    side = None if onset is None else onset.body_side
    side_scored = label.first_change_side not in (None, BodySide.UNCLEAR)
    return AngleScore(
        run=run,
        event_id=event.event_id,
        annotator=event.annotator,
        stage=stage,
        angle=clip.angle,
        onset_outcome=outcome,
        onset_label_s=label.onset_time_s,
        onset_predicted_s=predicted,
        onset_error_s=None if error is None else round(error, 6),
        onset_error_frames=None if error is None else round(error * clip.frame_rate, 3),
        contact_label=None if label.contact_status is None else label.contact_status.value,
        contact_predicted=None if contact is None else contact.value,
        contact_correct=None if label.contact_status is None else contact == label.contact_status,
        side_label=None if label.first_change_side is None else label.first_change_side.value,
        side_predicted=None if side is None else side.value,
        side_correct=side == label.first_change_side if side_scored else None,
    )


def _usage(run_dir: Path) -> dict:
    empty = {"model_calls": None, "input_tokens": None, "output_tokens": None, "model_latency_s": None}
    path = run_dir / "usage.jsonl"
    if not path.is_file():
        return empty
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def total(key: str):
        values = [row.get(key) for row in rows]
        return None if not values or any(value is None for value in values) else sum(values)

    return {
        "model_calls": len(rows),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "model_latency_s": None if not rows else round(sum(row["latency_s"] for row in rows), 3),
    }


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) or "unclear"


def _ratio(values) -> str:
    scored = [value for value in values if value is not None]
    if not scored:
        return "n/a"
    return f"{sum(1 for value in scored if value)}/{len(scored)}"


def _value(value) -> str:
    return "n/a" if value is None else str(value)
