from __future__ import annotations

from injury_analysis.constants import (
    BLIND_BLOCK,
    CANDIDATE_BLOCK,
    COMPARISON_BLOCK,
    OBSERVER_BLOCK,
    STAGE_BLIND,
    STAGE_CANDIDATE,
    STAGE_JUDGE,
    STAGE_OBSERVER,
)
from injury_analysis.errors import InjuryAnalysisError

_STEERING_PHRASES = (
    "caleb williams",
    "youtube.com",
    "-ysudwbehxk",
    "publicly reported",
)


def assert_safe_prompt(stage: str, system: str, user_text: str, video_name: str) -> None:
    combined = f"{system}\n{user_text}".lower()
    for phrase in _STEERING_PHRASES:
        if phrase in combined:
            raise InjuryAnalysisError(
                f"{stage} prompt contains steering text {phrase!r}. The model call was refused."
            )
    for name in [part.strip() for part in video_name.splitlines() if part.strip()]:
        if name.lower() in combined:
            raise InjuryAnalysisError(
                f"{stage} prompt contains the video file name. The model call was refused."
            )
    if stage == STAGE_OBSERVER:
        _reject_blocks(stage, user_text, (OBSERVER_BLOCK, CANDIDATE_BLOCK, BLIND_BLOCK, COMPARISON_BLOCK))
    elif stage == STAGE_BLIND:
        _reject_blocks(stage, user_text, (OBSERVER_BLOCK, CANDIDATE_BLOCK, BLIND_BLOCK, COMPARISON_BLOCK))
    elif stage == STAGE_CANDIDATE:
        _require_blocks(stage, user_text, (OBSERVER_BLOCK,))
        _reject_blocks(stage, user_text, (CANDIDATE_BLOCK, BLIND_BLOCK, COMPARISON_BLOCK))
    elif stage == STAGE_JUDGE:
        _require_blocks(stage, user_text, (OBSERVER_BLOCK, CANDIDATE_BLOCK, BLIND_BLOCK, COMPARISON_BLOCK))
    else:
        raise InjuryAnalysisError(f"Unknown analysis stage: {stage}")


def _require_blocks(stage: str, user_text: str, tags: tuple[str, ...]) -> None:
    for tag in tags:
        if f"<{tag}>" not in user_text:
            raise InjuryAnalysisError(f"{stage} prompt is missing <{tag}>")


def _reject_blocks(stage: str, user_text: str, tags: tuple[str, ...]) -> None:
    for tag in tags:
        if f"<{tag}>" in user_text:
            raise InjuryAnalysisError(
                f"{stage} prompt contains <{tag}>. That stage must not see this report."
            )
