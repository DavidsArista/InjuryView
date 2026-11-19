from conftest import ordered_stages
import json

import pytest

from injury_analysis.config import ClipSpec, RunConfig
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.mock_adapter import MockModelAdapter
from injury_analysis.models.schemas import ObserverArtifact
from injury_analysis.pipeline import run_pipeline
from injury_analysis.video.synthetic import write_synthetic_video


class FakeProvider:
    """Looks like a real provider to the pipeline. Delegates to the fixture model, with no network."""

    def __init__(self):
        self.inner = MockModelAdapter()

    @property
    def calls(self):
        return self.inner.calls

    def complete(self, request):
        return self.inner.complete(request)


def _config(video, output, **overrides) -> RunConfig:
    payload = dict(
        video_path=video,
        output_dir=output,
        broad_fps=4,
        focused_fps=12,
        mock=True,
        max_individual_images=40,
    )
    payload.update(overrides)
    return RunConfig(**payload)


def _stages(adapter) -> list[str]:
    return ordered_stages(adapter.calls)


@pytest.fixture
def two_videos(tmp_path):
    first = tmp_path / "clip_a.mp4"
    second = tmp_path / "clip_b.mp4"
    write_synthetic_video(first, duration_s=2, size="320x240")
    write_synthetic_video(second, duration_s=2, size="160x120")
    return first, second


def test_resume_refuses_frames_from_a_different_video(two_videos, tmp_path):
    first, second = two_videos
    output = tmp_path / "run"
    run_pipeline(_config(first, output), adapter=MockModelAdapter())
    saved_metadata = (output / "metadata.json").read_text()
    saved_usage = (output / "usage.jsonl").read_text()

    with pytest.raises(InjuryAnalysisError, match="not the file the saved frames came from"):
        run_pipeline(_config(second, output, resume=True), adapter=MockModelAdapter())
    assert (output / "metadata.json").read_text() == saved_metadata
    assert (output / "usage.jsonl").read_text() == saved_usage
    assert len(saved_usage.splitlines()) == 4


def test_resume_refuses_a_manifest_without_a_video_fingerprint(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output), adapter=MockModelAdapter())
    manifest = json.loads((output / "frame_manifest.json").read_text())
    manifest["video_sha256"] = None
    (output / "frame_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(InjuryAnalysisError, match="no video fingerprint"):
        run_pipeline(_config(synthetic_video, output, resume=True), adapter=MockModelAdapter())


def test_identical_resume_reuses_the_observer(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output), adapter=MockModelAdapter())
    again = MockModelAdapter()
    run_pipeline(_config(synthetic_video, output, resume=True), adapter=again)
    assert _stages(again) == ["candidate", "blind_reanalysis", "judge"]


def test_real_resume_does_not_reuse_mock_observer_output(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output), adapter=MockModelAdapter())

    real = FakeProvider()
    config = _config(
        synthetic_video,
        output,
        resume=True,
        mock=False,
        provider="muse",
        model_name="muse-spark-1.3",
    )
    run_pipeline(config, adapter=real)

    assert _stages(real)[0] == "observer"
    observer = ObserverArtifact.model_validate_json((output / "observer_report.json").read_text())
    assert observer.producer == "muse"
    assert observer.input_fingerprint


def test_saved_observer_without_a_fingerprint_is_not_reused(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output), adapter=MockModelAdapter())
    payload = json.loads((output / "observer_report.json").read_text())
    payload.pop("input_fingerprint")
    (output / "observer_report.json").write_text(json.dumps(payload))

    again = MockModelAdapter()
    run_pipeline(_config(synthetic_video, output, resume=True), adapter=again)
    assert _stages(again)[0] == "observer"


def test_changed_sheet_layout_rebuilds_sheets_and_reruns_the_observer(synthetic_video, tmp_path):
    output = tmp_path / "run"
    run_pipeline(_config(synthetic_video, output, max_individual_images=1), adapter=MockModelAdapter())
    before = sorted(path.name for path in (output / "contact_sheets").glob("*.jpg"))

    again = MockModelAdapter()
    run_pipeline(
        _config(synthetic_video, output, resume=True, max_individual_images=1, frames_per_sheet=2),
        adapter=again,
    )
    after = sorted(path.name for path in (output / "contact_sheets").glob("*.jpg"))
    manifest = json.loads((output / "frame_manifest.json").read_text())

    assert len(after) > len(before)
    assert manifest["frames_per_sheet"] == 2
    assert _stages(again)[0] == "observer"


def test_stale_prompt_logs_are_removed_and_invocations_are_logged(synthetic_video, tmp_path):
    output = tmp_path / "run"
    prompts = output / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "observer_retry1.txt").write_text("from an older invocation")

    run_pipeline(_config(synthetic_video, output), adapter=MockModelAdapter())
    run_pipeline(_config(synthetic_video, output, resume=True), adapter=MockModelAdapter())

    assert not (prompts / "observer_retry1.txt").exists()
    assert (prompts / "observer.txt").is_file()
    lines = [json.loads(line) for line in (output / "invocations.jsonl").read_text().splitlines()]
    assert [line["status"] for line in lines] == ["completed", "completed"]
    assert lines[0]["stages"]["observer"] == "ran"
    assert lines[1]["stages"]["observer"] == "reused"


def test_multi_angle_resume_refuses_a_swapped_video(two_videos, tmp_path):
    first, second = two_videos
    output = tmp_path / "angles"

    def config(wide, tight, **overrides):
        return _config(
            wide,
            output,
            clips=[
                ClipSpec(video_path=wide, label="wide", focus_start_s=0.5, focus_end_s=1.0),
                ClipSpec(video_path=tight, label="tight", focus_start_s=0.5, focus_end_s=1.0),
            ],
            **overrides,
        )

    run_pipeline(config(first, second), adapter=MockModelAdapter())
    same = MockModelAdapter()
    run_pipeline(config(first, second, resume=True), adapter=same)
    assert _stages(same) == ["candidate", "blind_reanalysis", "judge"]

    with pytest.raises(InjuryAnalysisError, match="Angle wide is not the file"):
        run_pipeline(config(second, first, resume=True), adapter=MockModelAdapter())
