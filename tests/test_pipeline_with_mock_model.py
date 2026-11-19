from conftest import calls_by_stage, ordered_stages
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from injury_analysis.cli import main
from injury_analysis.config import RunConfig
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.anthropic_adapter import AnthropicAdapter
from injury_analysis.models.openai_adapter import OpenAIAdapter
from injury_analysis.models.mock_adapter import (
    BLIND_PASS_MARKER,
    CANDIDATE_MARKER,
    FIRST_PASS_MARKER,
    MockModelAdapter,
)
from injury_analysis.models.model_adapter import ModelRequest
from injury_analysis.models.schemas import (
    CandidateArtifact,
    JudgeArtifact,
    MechanismCategory,
    ObserverArtifact,
    ObserverReport,
    RankingDecision,
)
from injury_analysis.pipeline import run_pipeline
from injury_analysis.video.frames import FrameManifest, FrameRecord

REQUIRED_ARTIFACTS = (
    "metadata.json",
    "frame_manifest.json",
    "observer_report.json",
    "candidate_report.json",
    "blind_reanalysis.json",
    "judge_report.json",
    "report.md",
    "run_config.json",
)


def _config(video, output, **overrides) -> RunConfig:
    payload = dict(
        video_path=video,
        output_dir=output,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.4,
        focus_end_s=1.2,
        source_offset_s=10,
        mock=True,
        max_individual_images=40,
    )
    payload.update(overrides)
    return RunConfig(**payload)


def test_mock_pipeline_writes_artifacts_and_keeps_passes_isolated(synthetic_video, tmp_path):
    output = tmp_path / "run"
    adapter = MockModelAdapter()
    result = run_pipeline(_config(synthetic_video, output), adapter=adapter)

    assert result.report_path.is_file()
    for name in REQUIRED_ARTIFACTS:
        assert (output / name).is_file()
    assert list((output / "frames").glob("*.jpg"))
    assert list((output / "contact_sheets").glob("*.jpg"))
    assert ordered_stages(adapter.calls) == [
        "observer",
        "candidate",
        "blind_reanalysis",
        "judge",
    ]

    calls = calls_by_stage(adapter.calls)
    observer_call, candidate_call, blind_call, judge_call = (
        calls["observer"], calls["candidate"], calls["blind_reanalysis"], calls["judge"]
    )
    assert blind_call.system == observer_call.system
    assert blind_call.user_text == observer_call.user_text
    assert FIRST_PASS_MARKER not in blind_call.user_text
    assert CANDIDATE_MARKER not in blind_call.user_text
    assert FIRST_PASS_MARKER in candidate_call.user_text
    assert BLIND_PASS_MARKER not in candidate_call.user_text
    assert FIRST_PASS_MARKER in judge_call.user_text
    assert BLIND_PASS_MARKER in judge_call.user_text
    for call in adapter.calls:
        assert synthetic_video.name not in f"{call.system}\n{call.user_text}"
        assert call.images

    manifest = FrameManifest.model_validate_json((output / "frame_manifest.json").read_text())
    observer = ObserverArtifact.model_validate_json((output / "observer_report.json").read_text())
    candidate = CandidateArtifact.model_validate_json((output / "candidate_report.json").read_text())
    blind = ObserverArtifact.model_validate_json((output / "blind_reanalysis.json").read_text())
    judge = JudgeArtifact.model_validate_json((output / "judge_report.json").read_text())
    report = (output / "report.md").read_text()

    assert observer.producer == "mock"
    assert observer.timestamp_authority == "video_presentation_timestamp"
    assert observer.timestamp_adjustments
    for observation in [*observer.observations, *blind.observations]:
        cited = [manifest.frame(frame_id).time_s for frame_id in observation.supporting_frame_ids]
        assert observation.start_time_s == pytest.approx(min(cited))
        assert observation.end_time_s == pytest.approx(max(cited))
    assert any(item.category == MechanismCategory.INSUFFICIENT for item in candidate.candidates)
    assert judge.ranking_decision == RankingDecision.INCONCLUSIVE
    assert judge.body_side_disagreements
    combined = "\n".join(
        [
            observer.model_dump_json(),
            candidate.model_dump_json(),
            blind.model_dump_json(),
            judge.model_dump_json(),
            report,
        ]
    ).lower()
    for phrase in ("hamstring", "achilles", "caleb", "youtube"):
        assert phrase not in combined
    assert "EXTERNAL GROUND TRUTH: none" in report
    assert "not a medical diagnosis" in report.lower()
    assert "(source " in report

    frame_bytes = {path.name: path.read_bytes() for path in (output / "frames").glob("*.jpg")}
    run_pipeline(_config(synthetic_video, output, resume=True), adapter=MockModelAdapter())
    assert {path.name: path.read_bytes() for path in (output / "frames").glob("*.jpg")} == frame_bytes


def test_subject_line_is_shared_and_a_changed_subject_reruns_the_observer(synthetic_video, tmp_path):
    subject = "QB, 18, Brown Jersey with White pants"
    output = tmp_path / "subject-run"
    adapter = MockModelAdapter()
    run_pipeline(_config(synthetic_video, output, subjects=[subject]), adapter=adapter)

    calls = calls_by_stage(adapter.calls)
    observer_call, candidate_call, blind_call, judge_call = (
        calls["observer"], calls["candidate"], calls["blind_reanalysis"], calls["judge"]
    )
    assert subject in observer_call.user_text
    assert "not a statement about contact" in observer_call.user_text
    assert observer_call.user_text == blind_call.user_text
    assert subject in candidate_call.user_text
    assert subject in judge_call.user_text
    report = (output / "report.md").read_text()
    assert f"**Subject to follow:** {subject}" in report
    assert "not an injury finding" in report
    assert "EXTERNAL GROUND TRUTH: none" in report

    same_subject = MockModelAdapter()
    run_pipeline(_config(synthetic_video, output, resume=True, subjects=[subject]), adapter=same_subject)
    assert ordered_stages(same_subject.calls) == ["candidate", "blind_reanalysis", "judge"]

    other = "Defender, white jersey, purple pants"
    changed = MockModelAdapter()
    run_pipeline(
        _config(synthetic_video, output, resume=True, subjects=[subject, other]),
        adapter=changed,
    )
    changed_calls = calls_by_stage(changed.calls)
    assert "observer" in changed_calls
    assert changed_calls["observer"].user_text == changed_calls["blind_reanalysis"].user_text
    assert "Player to follow:" in changed_calls["observer"].user_text
    assert other in changed_calls["observer"].user_text
    assert "not confused with the player to follow" in changed_calls["observer"].user_text


def test_subject_line_rejects_injury_language(synthetic_video, tmp_path):
    with pytest.raises(ValidationError):
        _config(synthetic_video, tmp_path / "bad", subjects=["the hamstring tore"])


def test_cli_mock_run_and_missing_inputs(synthetic_video, tmp_path, monkeypatch):
    output = tmp_path / "cli-run"
    code = main(
        [
            "analyze",
            "--video",
            str(synthetic_video),
            "--output",
            str(output),
            "--mock",
            "--broad-fps",
            "4",
            "--focused-fps",
            "12",
            "--focus-start",
            "0.4",
            "--focus-end",
            "1.2",
        ]
    )
    assert code == 0
    assert (output / "report.md").is_file()
    assert json.loads((output / "run_config.json").read_text())["status"] == "completed"

    assert main(["analyze", "--video", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "missing"), "--mock"]) == 2

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MODEL_API_KEY=fake-key-that-must-not-load\nANALYSIS_PROVIDER=muse\n")
    failed = tmp_path / "no-key"
    assert (
        main(
            [
                "analyze",
                "--video",
                str(synthetic_video),
                "--output",
                str(failed),
                "--broad-fps",
                "1",
            ]
        )
        == 2
    )
    assert (failed / "frame_manifest.json").is_file()
    assert not (failed / "report.md").exists()
    assert json.loads((failed / "run_config.json").read_text())["status"] == "failed"


def test_invalid_model_output_retries_once_then_fails(synthetic_video, tmp_path):
    class Flaky(MockModelAdapter):
        def __init__(self):
            super().__init__()
            self.observer_calls = 0

        def complete(self, request):
            if request.stage != "observer":
                return super().complete(request)
            self.observer_calls += 1
            report = super().complete(request)
            if self.observer_calls == 1:
                broken = report.observations[0].model_copy(
                    update={"supporting_frame_ids": ["missing-frame"]}
                )
                return report.model_copy(update={"observations": [broken, *report.observations[1:]]})
            return report

    flaky = Flaky()
    run_pipeline(_config(synthetic_video, tmp_path / "retry"), adapter=flaky)
    assert flaky.observer_calls == 2

    class AlwaysBad(MockModelAdapter):
        def complete(self, request):
            report = super().complete(request)
            if request.stage != "observer":
                return report
            broken = report.observations[0].model_copy(update={"supporting_frame_ids": ["missing-frame"]})
            return report.model_copy(update={"observations": [broken, *report.observations[1:]]})

    with pytest.raises(InjuryAnalysisError, match="unknown frame"):
        run_pipeline(_config(synthetic_video, tmp_path / "bad"), adapter=AlwaysBad())


def test_anthropic_adapter_sends_a_fresh_conversation_each_call():
    payload = {
        "summary": "Visible movement only.",
        "observations": [
            {
                "observation_id": "o001",
                "start_time_s": 0.0,
                "end_time_s": 0.0,
                "phase": "onset",
                "player": "unclear",
                "body_region": "lower limb",
                "body_side": "unclear",
                "visible_movement": "A change is visible.",
                "contact_status": "unclear",
                "supporting_frame_ids": ["f001"],
                "confidence": "low",
                "visibility_limitations": [],
                "statement_kind": "observed",
            }
        ],
        "phases_not_visible": [],
        "global_visibility_limitations": [],
    }
    messages = _FakeMessages(payload)
    adapter = AnthropicAdapter(model="test-model", api_key="not-used", client=SimpleNamespace(messages=messages))
    manifest = FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                path="frames/f001.jpg",
                width=8,
                height=8,
                sampling="broad",
            )
        ],
    )
    first = _request(manifest, "FIRST_ONLY_TEXT")
    second = _request(manifest, "SECOND_ONLY_TEXT")
    assert isinstance(adapter.complete(first), ObserverReport)
    assert isinstance(adapter.complete(second), ObserverReport)

    assert len(messages.calls) == 2
    for call in messages.calls:
        assert len(call["messages"]) == 1
        assert call["messages"][0]["role"] == "user"
        assert call["system"]
    first_body = json.dumps(messages.calls[0]["messages"])
    second_body = json.dumps(messages.calls[1]["messages"])
    assert "FIRST_ONLY_TEXT" in first_body
    assert "FIRST_ONLY_TEXT" not in second_body
    assert "SECOND_ONLY_TEXT" in second_body


def test_muse_adapter_requests_json_and_starts_a_fresh_conversation(tmp_path):
    from injury_analysis.models.model_adapter import ImagePayload
    from injury_analysis.models.muse_adapter import MuseAdapter

    image_path = tmp_path / "f001.jpg"
    image_path.write_bytes(b"not-a-real-jpeg")

    payload = {
        "summary": "Visible movement only.",
        "observations": [
            {
                "observation_id": "o001",
                "start_time_s": 0.0,
                "end_time_s": 0.0,
                "phase": "onset",
                "player": "unclear",
                "body_region": "lower limb",
                "body_side": "unclear",
                "visible_movement": "A change is visible.",
                "contact_status": "unclear",
                "supporting_frame_ids": ["f001"],
                "confidence": "low",
                "visibility_limitations": [],
                "statement_kind": "observed",
            }
        ],
        "phases_not_visible": [],
        "global_visibility_limitations": [],
    }
    completions = _FakeJsonCompletions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    adapter = MuseAdapter(model="muse-spark-1.3", api_key="not-used", client=client)
    manifest = FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                path="frames/f001.jpg",
                width=8,
                height=8,
                sampling="broad",
            )
        ],
    )
    image_payload = ImagePayload(relative_path="frames/f001.jpg", path=image_path, label="Frame f001")

    def request(user_text: str) -> ModelRequest:
        base = _request(manifest, user_text)
        return ModelRequest(
            stage=base.stage,
            system=base.system,
            user_text=base.user_text,
            images=[image_payload],
            response_model=base.response_model,
            manifest=manifest,
        )

    assert isinstance(adapter.complete(request("FIRST_ONLY_TEXT")), ObserverReport)
    assert isinstance(adapter.complete(request("SECOND_ONLY_TEXT")), ObserverReport)
    assert len(completions.calls) == 2
    for call in completions.calls:
        assert call["model"] == "muse-spark-1.3"
        assert "tools" not in call
        assert call["response_format"]["type"] == "json_schema"
        assert len(call["messages"]) == 2
        image = call["messages"][1]["content"][1]["image_url"]
        assert "detail" not in image
        assert image["url"].startswith("data:image/jpeg;base64,")
    first_body = json.dumps(completions.calls[0]["messages"])
    second_body = json.dumps(completions.calls[1]["messages"])
    assert "FIRST_ONLY_TEXT" in first_body
    assert "FIRST_ONLY_TEXT" not in second_body


def test_muse_client_uses_the_meta_model_endpoint(monkeypatch):
    from injury_analysis.models.muse_adapter import MuseAdapter

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    MuseAdapter(model="muse-spark-1.3", api_key="muse-key")._client()
    assert captured["api_key"] == "muse-key"
    assert captured["base_url"] == "https://api.meta.ai/v1"


def test_openai_adapter_sends_a_fresh_conversation_each_call():
    payload = {
        "summary": "Visible movement only.",
        "observations": [
            {
                "observation_id": "o001",
                "start_time_s": 0.0,
                "end_time_s": 0.0,
                "phase": "onset",
                "player": "unclear",
                "body_region": "lower limb",
                "body_side": "unclear",
                "visible_movement": "A change is visible.",
                "contact_status": "unclear",
                "supporting_frame_ids": ["f001"],
                "confidence": "low",
                "visibility_limitations": [],
                "statement_kind": "observed",
            }
        ],
        "phases_not_visible": [],
        "global_visibility_limitations": [],
    }
    completions = _FakeCompletions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    adapter = OpenAIAdapter(model="gpt-4.1", api_key="not-used", client=client)
    manifest = FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                path="frames/f001.jpg",
                width=8,
                height=8,
                sampling="broad",
            )
        ],
    )
    assert isinstance(adapter.complete(_request(manifest, "FIRST_ONLY_TEXT")), ObserverReport)
    assert isinstance(adapter.complete(_request(manifest, "SECOND_ONLY_TEXT")), ObserverReport)
    assert len(completions.calls) == 2
    for call in completions.calls:
        assert len(call["messages"]) == 2
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][1]["role"] == "user"
    first_body = json.dumps(completions.calls[0]["messages"])
    second_body = json.dumps(completions.calls[1]["messages"])
    assert "FIRST_ONLY_TEXT" in first_body
    assert "FIRST_ONLY_TEXT" not in second_body
    assert "SECOND_ONLY_TEXT" in second_body


class _FakeJsonCompletions:
    def __init__(self, payload, usage=None):
        self.payload = payload
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload))
        choice = SimpleNamespace(finish_reason="stop", message=message)
        return SimpleNamespace(choices=[choice], usage=self.usage)


class _FakeCompletions:
    def __init__(self, payload, usage=None):
        self.payload = payload
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        function = SimpleNamespace(name="submit_analysis", arguments=json.dumps(self.payload))
        message = SimpleNamespace(tool_calls=[SimpleNamespace(function=function)])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=self.usage)


class _FakeMessages:
    def __init__(self, payload, usage=None):
        self.payload = payload
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", name="submit_analysis", input=self.payload)
        return SimpleNamespace(stop_reason="tool_use", content=[block], usage=self.usage)


def _request(manifest: FrameManifest, user_text: str) -> ModelRequest:
    return ModelRequest(
        stage="observer",
        system="Describe visible movement only.",
        user_text=user_text,
        images=[],
        response_model=ObserverReport,
        manifest=manifest,
    )


def test_usage_recorder_captures_provider_token_counts():
    from injury_analysis.models.model_adapter import UsageRecorder
    from injury_analysis.models.muse_adapter import MuseAdapter

    payload = {
        "summary": "Visible movement only.",
        "observations": [
            {
                "observation_id": "o001",
                "start_time_s": 0.0,
                "end_time_s": 0.0,
                "phase": "onset",
                "player": "unclear",
                "body_region": "lower limb",
                "body_side": "unclear",
                "visible_movement": "A change is visible.",
                "contact_status": "unclear",
                "supporting_frame_ids": ["f001"],
                "confidence": "low",
                "visibility_limitations": [],
                "statement_kind": "observed",
            }
        ],
        "phases_not_visible": [],
        "global_visibility_limitations": [],
    }
    manifest = FrameManifest(
        broad_fps=4,
        focused_fps=12,
        frames=[
            FrameRecord(
                frame_id="f001",
                source_frame_index=0,
                pts_s=0.0,
                time_s=0.0,
                path="frames/f001.jpg",
                width=8,
                height=8,
                sampling="broad",
            )
        ],
    )
    usage = SimpleNamespace(input_tokens=120, output_tokens=30, prompt_tokens=120, completion_tokens=30)
    anthropic = AnthropicAdapter(model="m", api_key="not-used", client=SimpleNamespace(messages=_FakeMessages(payload, usage)))
    openai = OpenAIAdapter(
        model="m",
        api_key="not-used",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(payload, usage))),
    )
    muse = MuseAdapter(
        model="m",
        api_key="not-used",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_FakeJsonCompletions(payload, usage))),
    )
    for adapter in (anthropic, openai, muse):
        recorder = UsageRecorder(adapter)
        recorder.complete(_request(manifest, "TEXT"))
        recorder.complete(_request(manifest, "TEXT"))
        first, second = recorder.records
        assert (first["input_tokens"], first["output_tokens"]) == (120, 30)
        assert (first["attempt"], second["attempt"]) == (1, 2)
        assert first["latency_s"] >= 0 and first["error"] is None
