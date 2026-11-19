"""The judge receives the pass comparison from compare.py instead of producing it again."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from injury_analysis.constants import COMPARISON_BLOCK, STAGE_BLIND, STAGE_CANDIDATE, STAGE_JUDGE, STAGE_OBSERVER
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.schemas import JudgeArtifact, JudgeReport
from injury_analysis.stages.isolation import assert_safe_prompt

_JUDGE = {
    "claim_assessments": [
        {
            "claim_id": "c001",
            "claim_text": "A visible change.",
            "verdict": "partially_supported",
            "rationale": "Seen in one pass.",
            "frame_ids": ["f001"],
            "start_time_s": 0.0,
            "end_time_s": 0.0,
        }
    ],
    "ranking_decision": "inconclusive",
    "ranking_rationale": "Not enough to separate the candidates.",
}
_LEGACY_FIELDS = {
    "reproduced_by_both_passes": ["o001 and b001"],
    "only_in_first_pass": [],
    "only_in_second_pass": [],
    "body_side_disagreements": ["o002 left vs b002 right"],
}


def test_model_output_may_not_include_the_comparison_fields():
    # The model is not asked for these any more; if it adds them, the schema rejects them.
    with pytest.raises(ValidationError):
        JudgeReport.model_validate({**_JUDGE, **_LEGACY_FIELDS})


def test_judge_artifacts_from_older_runs_still_load():
    legacy = {
        **_JUDGE,
        **_LEGACY_FIELDS,
        "producer": "openai",
        "model_name": "gpt-4.1",
        "timestamp_authority": "video_presentation_timestamp",
    }
    artifact = JudgeArtifact.model_validate_json(json.dumps(legacy))
    assert artifact.ranking_decision.value == "inconclusive"


def _with_blocks(*tags: str) -> str:
    return "\n".join(f"<{tag}>\n{{}}\n</{tag}>" for tag in tags)


@pytest.mark.parametrize("stage", [STAGE_OBSERVER, STAGE_BLIND])
def test_observation_passes_never_see_the_comparison(stage):
    with pytest.raises(InjuryAnalysisError):
        assert_safe_prompt(stage, "system", _with_blocks(COMPARISON_BLOCK), "clip.mp4")


def test_candidate_stage_never_sees_the_comparison():
    text = _with_blocks("observer_report_json", COMPARISON_BLOCK)
    with pytest.raises(InjuryAnalysisError):
        assert_safe_prompt(STAGE_CANDIDATE, "system", text, "clip.mp4")


def test_judge_prompt_requires_the_comparison():
    without = _with_blocks("observer_report_json", "candidate_report_json", "blind_reanalysis_json")
    with pytest.raises(InjuryAnalysisError, match=COMPARISON_BLOCK):
        assert_safe_prompt(STAGE_JUDGE, "system", without, "clip.mp4")
    assert_safe_prompt(STAGE_JUDGE, "system", without + "\n" + _with_blocks(COMPARISON_BLOCK), "clip.mp4")
