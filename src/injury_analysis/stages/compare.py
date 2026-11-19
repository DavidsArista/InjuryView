from __future__ import annotations

import re
from dataclasses import dataclass

from injury_analysis.models.schemas import BodySide, Observation, ObserverReport
from injury_analysis.video.frames import FrameManifest

_SIDED = {BodySide.LEFT, BodySide.RIGHT}
_UNCLEAR = "unclear"


@dataclass(frozen=True)
class PassComparison:
    body_side_disagreements: list[str]
    only_in_first: list[str]
    only_in_second: list[str]
    tentative_pairs: list[str]


def compare_observations(
    first: ObserverReport,
    second: ObserverReport,
    manifest: FrameManifest,
) -> PassComparison:
    """Pair observations across the two passes.

    A pair needs the same phase, the same camera angle, a compatible player, and overlapping
    times on that angle's clock. Times on different angles are never compared.
    """
    disagreements: list[str] = []
    only_first: list[str] = []
    tentative: list[str] = []
    used_second: set[int] = set()
    for left in first.observations:
        partner_index = _partner_index(left, second.observations, used_second, manifest)
        if partner_index is None:
            only_first.append(_describe(left, manifest))
            continue
        used_second.add(partner_index)
        right = second.observations[partner_index]
        is_tentative = _player(left) != _player(right)
        if is_tentative:
            tentative.append(
                f"{left.observation_id} ({left.player}) paired with {right.observation_id} "
                f"({right.player}) on {_angle_label(left, manifest)}; one pass did not name the player."
            )
        if left.body_side in _SIDED and right.body_side in _SIDED and left.body_side != right.body_side:
            disagreements.append(
                f"{left.observation_id} says {left.body_side.value} and {right.observation_id} "
                f"says {right.body_side.value} for phase {left.phase.value} "
                f"on {_angle_label(left, manifest)} "
                f"at {left.start_time_s:.2f}-{left.end_time_s:.2f}s "
                f"(frames {', '.join(left.supporting_frame_ids)} vs {', '.join(right.supporting_frame_ids)})"
                + (". Tentative pair." if is_tentative else ".")
            )
    only_second = [
        _describe(observation, manifest)
        for index, observation in enumerate(second.observations)
        if index not in used_second
    ]
    return PassComparison(
        body_side_disagreements=disagreements,
        only_in_first=only_first,
        only_in_second=only_second,
        tentative_pairs=tentative,
    )


def _partner_index(
    left: Observation,
    others: list[Observation],
    used: set[int],
    manifest: FrameManifest,
) -> int | None:
    best: tuple[float, int] | None = None
    left_angle = _angle(left, manifest)
    left_player = _player(left)
    for index, right in enumerate(others):
        if index in used:
            continue
        if left.phase != right.phase or _angle(right, manifest) != left_angle:
            continue
        right_player = _player(right)
        if left_player != right_player and _UNCLEAR not in (left_player, right_player):
            continue
        overlap = min(left.end_time_s, right.end_time_s) - max(left.start_time_s, right.start_time_s)
        if overlap < 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, index)
    return None if best is None else best[1]


def _angle(observation: Observation, manifest: FrameManifest) -> frozenset[str]:
    frames = manifest.by_id()
    return frozenset(
        frames[frame_id].clip_id for frame_id in observation.supporting_frame_ids if frame_id in frames
    )


def _angle_label(observation: Observation, manifest: FrameManifest) -> str:
    angles = sorted(angle for angle in _angle(observation, manifest) if angle)
    return ", ".join(angles) if angles else "the clip"


def _player(observation: Observation) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", observation.player.lower()).split()) or _UNCLEAR


def _describe(observation: Observation, manifest: FrameManifest) -> str:
    movement = " ".join(observation.visible_movement.split())
    if len(movement) > 180:
        movement = movement[:177] + "..."
    angle = _angle_label(observation, manifest)
    where = "" if angle == "the clip" else f"angle={angle} "
    return (
        f"{observation.observation_id} phase={observation.phase.value} "
        f"side={observation.body_side.value} {where}"
        f"{observation.start_time_s:.2f}-{observation.end_time_s:.2f}s "
        f"frames={','.join(observation.supporting_frame_ids)}: {movement}"
    )
