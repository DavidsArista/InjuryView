from __future__ import annotations

from injury_analysis.constants import DISCLAIMER
from injury_analysis.models.schemas import (
    CandidateArtifact,
    Confidence,
    JudgeArtifact,
    MechanismCategory,
    ObserverArtifact,
    StatementKind,
)
from injury_analysis.stages.compare import PassComparison, compare_observations
from injury_analysis.video.frames import FrameManifest
from injury_analysis.video.metadata import VideoMetadata


def render_report(
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverArtifact,
    candidate: CandidateArtifact,
    blind: ObserverArtifact,
    judge: JudgeArtifact,
    subjects: list[str] | None = None,
    stage_frames: dict[str, int] | None = None,
) -> str:
    subject_lines = subjects or []
    comparison = compare_observations(observer, blind, manifest)
    offset = metadata.source_offset_s
    sections = [
        _header(observer),
        _metadata_section(metadata, manifest, subject_lines),
        _timeline_section(observer, manifest, offset),
        _observation_section(observer, blind, manifest, offset),
        _candidate_section(candidate, manifest, offset),
        _evidence_section(candidate, manifest, offset),
        _review_section(comparison, judge, manifest, offset, subject_lines),
        _disputed_section(judge),
        _conclusion_section(candidate, judge, manifest, offset),
        _limitations_section(metadata, manifest, observer, blind, judge, stage_frames or {}),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _header(observer: ObserverArtifact) -> str:
    lines = [f"> {DISCLAIMER}"]
    if observer.producer == "mock":
        lines.append(
            "> Analysis source: mock fixture. The observations below were not produced by a "
            "vision model and are not a reading of the video."
        )
    return "\n".join(lines)


def _metadata_section(metadata: VideoMetadata, manifest: FrameManifest, subjects: list[str]) -> str:
    focus = "not configured"
    if manifest.focus_start_s is not None and manifest.focus_end_s is not None:
        focus = f"{manifest.focus_start_s:.2f}–{manifest.focus_end_s:.2f}s"
    offset = "not supplied" if metadata.source_offset_s is None else f"{metadata.source_offset_s:.3f}s"
    rows = [
        ("File", metadata.video_path),
        ("Duration", f"{metadata.duration_s:.3f}s"),
        ("Resolution", f"{metadata.width}x{metadata.height}"),
        ("Frame rate", f"{metadata.frame_rate:.6f} ({metadata.frame_rate_raw})"),
        ("Time base", metadata.time_base),
        ("Codec", metadata.codec_name),
        ("Container start", f"{metadata.container_start_s:.6f}s"),
        ("Probed frames", str(metadata.probed_frame_count)),
        ("Extracted frames", str(len(manifest.frames))),
        ("Broad sampling", f"{manifest.broad_fps:g} fps"),
        ("Focused sampling", f"{manifest.focused_fps:g} fps"),
        ("Focused window", focus),
        ("Source offset", offset),
        ("Subject to follow", _subject_label(subjects)),
        ("Timestamp authority", "presentation timestamps of the cited frames, relative to this clip"),
    ]
    lines = ["## 1. Video metadata", ""]
    if len(manifest.clips) > 1:
        lines.append("These angles were analyzed together. Each angle has its own clock.")
        lines.append("")
        for clip in manifest.clips:
            window = "not configured"
            if clip.focus_start_s is not None and clip.focus_end_s is not None:
                window = f"{clip.focus_start_s:.2f}–{clip.focus_end_s:.2f}s"
            lines.append(f"- **{clip.clip_id}:** {clip.video_path}")
            lines.append(f"  - Duration: {clip.duration_s:.3f}s")
            lines.append(f"  - Frame rate: {clip.frame_rate:.6f}")
            lines.append(f"  - Focused window: {window}")
        lines.append(f"- **Extracted frames:** {len(manifest.frames)}")
        lines.append(f"- **Broad sampling:** {manifest.broad_fps:g} fps")
        lines.append(f"- **Focused sampling:** {manifest.focused_fps:g} fps")
        lines.append("- **Timestamp authority:** presentation timestamps of the cited frames, on that angle's own clock")
        return "\n".join(lines)
    for label, value in rows:
        lines.append(f"- **{label}:** {value}")
    return "\n".join(lines)


def _timeline_section(observer: ObserverArtifact, manifest: FrameManifest, offset: float | None) -> str:
    lines = [
        "## 2. Timestamped movement timeline",
        "",
        "| Time | Phase | Label | Body side | Movement | Frames |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for observation in sorted(observer.observations, key=lambda item: (item.start_time_s, item.observation_id)):
        lines.append(
            "| {time} | {phase} | {label} | {side} | {movement} | {frames} |".format(
                time=_span(observation.start_time_s, observation.end_time_s, offset),
                phase=observation.phase.value,
                label=_kind(observation.statement_kind),
                side=observation.body_side.value,
                movement=_cell(observation.visible_movement),
                frames=_cell(", ".join(observation.supporting_frame_ids)),
            )
        )
    return "\n".join(lines)


def _observation_section(
    observer: ObserverArtifact,
    blind: ObserverArtifact,
    manifest: FrameManifest,
    offset: float | None,
) -> str:
    lines = ["## 3. Contact and movement observations", ""]
    lines.append("### First observer pass")
    lines.append("")
    lines.extend(_observation_bullets(observer, manifest, offset))
    lines.append("")
    lines.append("### Blind second observer pass")
    lines.append("")
    lines.extend(_observation_bullets(blind, manifest, offset))
    return "\n".join(lines)


def _observation_bullets(report: ObserverArtifact, manifest: FrameManifest, offset: float | None) -> list[str]:
    lines = [
        f"Producer: {report.producer} ({report.model_name}).",
        "",
        report.summary,
        "",
    ]
    for observation in report.observations:
        citations = _cite(observation.supporting_frame_ids, manifest, offset)
        limitations = "; ".join(item.rstrip(".") for item in observation.visibility_limitations) or "none recorded"
        lines.append(
            f"- `{observation.observation_id}` {_kind(observation.statement_kind)} "
            f"{_span(observation.start_time_s, observation.end_time_s, offset)} "
            f"phase `{observation.phase.value}`, {observation.player}, "
            f"{observation.body_region} ({observation.body_side.value}). "
            f"{observation.visible_movement} "
            f"Contact: `{observation.contact_status.value}`. "
            f"Confidence: {observation.confidence.value}. "
            f"Limitations: {limitations}. Frames: {citations}."
        )
    if report.phases_not_visible:
        missing = ", ".join(phase.value for phase in report.phases_not_visible)
        lines.append("")
        lines.append(f"Phases not covered by this pass: {missing}.")
    return lines


def _candidate_section(candidate: CandidateArtifact, manifest: FrameManifest, offset: float | None) -> str:
    lines = [
        "## 4. Possible injury mechanisms",
        "",
        "INFERRED. These are visible-mechanism hypotheses. They are not a diagnosis.",
        "",
        f"Producer: {candidate.producer} ({candidate.model_name}).",
        "",
        candidate.summary,
        "",
    ]
    for rank, item in enumerate(candidate.candidates, start=1):
        lines.append(
            f"{rank}. **{item.name}** (`{item.category.value}`, confidence {item.confidence.value}). "
            f"{item.mechanism_description} "
            f"Observations: {_id_list(item.observation_ids)}. "
            f"Frames: {_cite(item.frame_ids, manifest, offset) or 'none cited'}."
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _evidence_section(candidate: CandidateArtifact, manifest: FrameManifest, offset: float | None) -> str:
    lines = ["## 5. Evidence supporting and weakening each possibility", ""]
    for item in candidate.candidates:
        lines.append(f"### {item.candidate_id}: {item.name}")
        lines.append("")
        lines.append("Supporting evidence:")
        lines.extend(_evidence_lines(item.supporting_evidence, manifest, offset) or ["- None recorded."])
        lines.append("")
        lines.append("Weakening or contradictory evidence:")
        lines.extend(_evidence_lines(item.weakening_evidence, manifest, offset) or ["- None recorded."])
        lines.append("")
        lines.append("Evidence that would distinguish this possibility:")
        for needed in item.distinguishing_evidence_needed:
            lines.append(f"- {needed}")
        lines.append("")
        lines.append("Unsupported assumptions:")
        if item.unsupported_assumptions:
            for assumption in item.unsupported_assumptions:
                lines.append(f"- {assumption}")
        else:
            lines.append("- None recorded.")
        lines.append("")
    return "\n".join(lines).rstrip()


def _subject_label(subjects: list[str]) -> str:
    if not subjects:
        return "not supplied"
    return "; ".join(subjects)


def _subject_note(subjects: list[str]) -> str:
    if not subjects:
        return ""
    listed = "; ".join(subjects)
    return (
        f"Subject identity was supplied so the model would follow one player: {listed}. "
        "That description is not an injury finding.\n\n"
    )


def _review_section(
    comparison: PassComparison,
    judge: JudgeArtifact,
    manifest: FrameManifest,
    offset: float | None,
    subjects: list[str],
) -> str:
    lines = [
        "## 6. Independent-review findings",
        "",
        _subject_note(subjects)
        + "EXTERNAL GROUND TRUTH: none. No external ground truth was provided to the observer, "
        "the candidate analyzer, the blind reanalysis, or the judge.",
        "",
        "### Deterministic field comparison",
        "",
        "This comparison uses the structured fields after both passes were bound to presentation timestamps. "
        "It pairs observations only within one camera angle and for the same player. "
        "It does not force the two passes to agree.",
        "",
    ]
    lines.extend(_bullets("Reproduced by both passes", comparison.paired))
    lines.extend(_bullets("Body-side disagreements", comparison.body_side_disagreements))
    lines.extend(_bullets("Only in the first pass", comparison.only_in_first))
    lines.extend(_bullets("Only in the second pass", comparison.only_in_second))
    lines.extend(_bullets("Tentative pairs (one pass did not name the player)", comparison.tentative_pairs))
    lines.append("### Judge assessment")
    lines.append("")
    lines.append(
        f"INFERRED. Producer: {judge.producer} ({judge.model_name}). "
        f"Ranking decision: `{judge.ranking_decision.value}`."
    )
    lines.append("")
    lines.append(judge.ranking_rationale)
    lines.append("")
    lines.extend(_bullets("Timestamp disagreements that affect a claim", judge.timestamp_disagreements))
    lines.extend(_bullets("Missing plausible alternatives", judge.missing_plausible_alternatives))
    lines.append("Claim classifications:")
    lines.append("")
    for claim in judge.claim_assessments:
        lines.append(
            f"- `{claim.claim_id}` `{claim.verdict.value}` "
            f"{_span(claim.start_time_s, claim.end_time_s, offset)}. "
            f"{claim.claim_text} {claim.rationale} "
            f"Frames: {_cite(claim.frame_ids, manifest, offset)}. "
            f"Observations: {_id_list(claim.related_observation_ids)}."
        )
    return "\n".join(lines)


def _disputed_section(judge: JudgeArtifact) -> str:
    lines = ["## 7. Unsupported or disputed claims", ""]
    if not judge.unsupported_claims:
        lines.append("None recorded.")
        return "\n".join(lines)
    disputed = [
        claim
        for claim in judge.claim_assessments
        if claim.verdict.value in {"contradicted", "not_visible", "interpretation_presented_as_observation"}
    ]
    for item in judge.unsupported_claims:
        lines.append(f"- {item}")
    if disputed:
        lines.append("")
        lines.append("Claims the judge did not treat as directly supported:")
        for claim in disputed:
            lines.append(f"- `{claim.claim_id}` `{claim.verdict.value}`: {claim.claim_text}")
    return "\n".join(lines)


def _conclusion_section(candidate: CandidateArtifact, judge: JudgeArtifact, manifest: FrameManifest, offset: float | None) -> str:
    leading = candidate.candidates[0]
    decision = {
        "retain": "The judge would retain the candidate ranking.",
        "change": "The judge would change the candidate ranking.",
        "inconclusive": "The judge marked the ranking inconclusive.",
    }[judge.ranking_decision.value]
    lines = [
        "## 8. Overall conclusion",
        "",
        "INFERRED.",
        "",
        decision,
        "",
        judge.ranking_rationale,
        "",
        (
            f"The leading candidate in the submitted ranking is {leading.name} "
            f"(`{leading.category.value}`, confidence {leading.confidence.value}) "
            f"at {_cite(leading.frame_ids, manifest, offset) or 'no cited frames'}."
        ),
        "",
        DISCLAIMER,
        "",
        "EXTERNAL GROUND TRUTH: none supplied.",
    ]
    if leading.category == MechanismCategory.INSUFFICIENT or leading.confidence == Confidence.LOW:
        lines.extend(["", "Uncertainty is high enough that no single mechanism should be treated as established."])
    return "\n".join(lines)


def _limitations_section(
    metadata: VideoMetadata,
    manifest: FrameManifest,
    observer: ObserverArtifact,
    blind: ObserverArtifact,
    judge: JudgeArtifact,
    stage_frames: dict[str, int],
) -> str:
    lines = [
        "## 9. Important limitations",
        "",
        "- A single camera angle, occlusion, motion blur, and resolution can hide contact and limb position.",
        "- Timestamps in this report are clip-relative presentation timestamps of the cited frames.",
        "- The model does not choose those timestamps. If it proposes different numbers, the report replaces them.",
        "- Sampled frames can miss motion between samples.",
        "- Contact sheets are reduced copies for review. The original extracted frames are kept.",
        "- No external ground truth was supplied to any analysis stage.",
        "- Video cannot establish tissue damage, a grade, or severity.",
    ]
    if metadata.source_offset_s is not None:
        lines.append("- Source times are the clip-relative time plus the operator-supplied offset.")
    total = len(manifest.frames)
    for stage, shown in stage_frames.items():
        if shown < total:
            lines.append(
                f"- The {stage} stage saw {shown} of {total} extracted frames: the frames cited by earlier passes. "
                "It could not check frames that no earlier pass cited."
            )
    if observer.producer == "mock":
        lines.append("- This run used the mock fixture model. It did not look at the pixels.")
    for note in observer.global_visibility_limitations:
        lines.append(f"- First pass limitation: {note}")
    for note in blind.global_visibility_limitations:
        lines.append(f"- Blind pass limitation: {note}")
    for note in judge.limitations:
        lines.append(f"- Judge limitation: {note}")
    for note in observer.timestamp_adjustments:
        lines.append(f"- Timestamp binding (first pass): {note}")
    for note in blind.timestamp_adjustments:
        lines.append(f"- Timestamp binding (blind pass): {note}")
    for note in judge.timestamp_adjustments:
        lines.append(f"- Timestamp binding (judge): {note}")
    return "\n".join(lines)


def _evidence_lines(links, manifest: FrameManifest, offset: float | None) -> list[str]:
    return [
        f"- `{link.observation_id}`: {link.note} Frames: {_cite(link.frame_ids, manifest, offset)}."
        for link in links
    ]


def _bullets(title: str, items: list[str]) -> list[str]:
    lines = [f"{title}:", ""]
    if not items:
        lines.append("- None recorded.")
    else:
        lines.extend(f"- {item}" for item in items)
    lines.append("")
    return lines


def _kind(kind: StatementKind) -> str:
    if kind == StatementKind.OBSERVED:
        return "OBSERVED"
    return "INFERRED"


def _span(start: float, end: float, offset: float | None) -> str:
    clip = _range(start, end)
    if offset is None:
        return clip
    return f"{clip} (source {_range(start + offset, end + offset)})"


def _range(start: float, end: float) -> str:
    if abs(end - start) < 0.005:
        return f"{start:.2f}s"
    return f"{start:.2f}–{end:.2f}s"


def _cite(frame_ids: list[str], manifest: FrameManifest, offset: float | None) -> str:
    parts = []
    for frame_id in frame_ids:
        frame = manifest.frame(frame_id)
        clock = _span(frame.time_s, frame.time_s, offset)
        if frame.clip_id:
            clock = f"{frame.clip_id} {clock}"
        parts.append(f"{frame.frame_id} ({clock})")
    return ", ".join(parts)


def _id_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "/")
