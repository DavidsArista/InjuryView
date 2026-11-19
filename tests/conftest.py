import shutil
import socket

import pytest

from injury_analysis.video.synthetic import write_synthetic_video

_PROVIDER_VARIABLES = (
    "MODEL_API_KEY",
    "MUSE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANALYSIS_PROVIDER",
    "MUSE_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_from_credentials_and_network(monkeypatch):
    """Tests never read the project .env, never see provider keys, and never open a socket."""
    for name in _PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for target in (
        "injury_analysis.config.load_project_env",
        "injury_analysis.cli.load_project_env",
        "injury_analysis.pipeline.load_project_env",
    ):
        monkeypatch.setattr(target, lambda: None)

    def _refuse(*args, **kwargs):
        raise RuntimeError("Network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path_factory.mktemp("media") / "reference_clip_do_not_leak.mp4"
    write_synthetic_video(path, duration_s=2, fps="30", size="320x240")
    return path


# The blind pass runs alongside observer -> candidate, so the order of model calls is not
# fixed. Tests compare stages in pipeline order and look calls up by stage.
STAGE_ORDER = ("observer", "candidate", "blind_reanalysis", "judge")


def ordered_stages(calls) -> list[str]:
    return sorted((call.stage for call in calls), key=STAGE_ORDER.index)


def calls_by_stage(calls) -> dict:
    by_stage = {}
    for call in calls:
        assert call.stage not in by_stage, f"{call.stage} was called more than once"
        by_stage[call.stage] = call
    return by_stage
