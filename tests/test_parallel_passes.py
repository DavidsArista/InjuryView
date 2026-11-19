"""The blind pass runs alongside observer -> candidate without weakening isolation or usage logs."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from conftest import calls_by_stage, ordered_stages
from injury_analysis.config import RunConfig
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.model_adapter import UsageRecorder, UsageTracking
from injury_analysis.pipeline import run_pipeline

FIRST_PASS_MARKER = "<observer_report_json>"


def _config(video, output, **overrides) -> RunConfig:
    payload = dict(
        video_path=video,
        output_dir=output,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.4,
        focus_end_s=1.2,
        mock=True,
        max_individual_images=40,
    )
    payload.update(overrides)
    return RunConfig(**payload)


class _RendezvousAdapter(MockModelAdapter):
    """Observer and blind calls each wait until the other has started.

    If the pipeline ran them one after the other, the first would time out.
    """

    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(2, timeout=10)

    def complete(self, request):
        if request.stage in {"observer", "blind_reanalysis"}:
            self.barrier.wait()
        return super().complete(request)


def test_observer_and_blind_pass_run_at_the_same_time(synthetic_video, tmp_path):
    adapter = _RendezvousAdapter()
    run_pipeline(_config(synthetic_video, tmp_path / "run"), adapter=adapter)

    calls = calls_by_stage(adapter.calls)
    assert ordered_stages(adapter.calls) == ["observer", "candidate", "blind_reanalysis", "judge"]
    # Running early does not let the blind pass see the first report.
    assert calls["blind_reanalysis"].user_text == calls["observer"].user_text
    assert FIRST_PASS_MARKER not in calls["blind_reanalysis"].user_text
    assert (tmp_path / "run" / "blind_reanalysis.json").is_file()


def test_sequential_flag_keeps_the_original_order(synthetic_video, tmp_path):
    adapter = MockModelAdapter()
    run_pipeline(_config(synthetic_video, tmp_path / "run", parallel_passes=False), adapter=adapter)
    assert [call.stage for call in adapter.calls] == ["observer", "candidate", "blind_reanalysis", "judge"]


def test_a_failing_candidate_stage_still_fails_the_run(synthetic_video, tmp_path):
    class FailingCandidate(MockModelAdapter):
        def complete(self, request):
            if request.stage == "candidate":
                raise InjuryAnalysisError("candidate stage failed")
            return super().complete(request)

    with pytest.raises(InjuryAnalysisError, match="candidate stage failed"):
        run_pipeline(_config(synthetic_video, tmp_path / "run"), adapter=FailingCandidate())
    config = (tmp_path / "run" / "run_config.json").read_text()
    assert '"status": "failed"' in config


def test_concurrent_calls_record_their_own_token_counts():
    class SlowAdapter(UsageTracking):
        """Sets its usage, then waits so the other thread can overwrite a shared attribute."""

        def __init__(self):
            self.barrier = threading.Barrier(2, timeout=10)

        def complete(self, request):
            tokens = 100 if request.stage == "observer" else 7
            self.last_usage = {"input_tokens": tokens, "output_tokens": tokens}
            self.barrier.wait()
            return SimpleNamespace()

    recorder = UsageRecorder(SlowAdapter())
    requests = [SimpleNamespace(stage=stage, images=[]) for stage in ("observer", "blind_reanalysis")]
    threads = [threading.Thread(target=recorder.complete, args=(request,)) for request in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tokens = {row["stage"]: row["input_tokens"] for row in recorder.records}
    assert tokens == {"observer": 100, "blind_reanalysis": 7}
