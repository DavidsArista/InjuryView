from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from injury_analysis.constants import DEFAULT_MODEL, DEFAULT_MUSE_MODEL, DEFAULT_OPENAI_MODEL
from injury_analysis.errors import InjuryAnalysisError


class ClipSpec(BaseModel):
    """One local camera angle. The label is sent to the model. The path is not."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    video_path: Path
    label: str
    focus_start_s: float | None = None
    focus_end_s: float | None = None

    @model_validator(mode="after")
    def _check_clip(self) -> "ClipSpec":
        label = self.label.strip()
        if not label or "/" in label or "\\" in label:
            raise ValueError("A clip label must be a short angle name, not a file path")
        if (self.focus_start_s is None) ^ (self.focus_end_s is None):
            raise ValueError(f"{label} needs both focus times, or neither")
        if (
            self.focus_start_s is not None
            and self.focus_end_s is not None
            and self.focus_end_s <= self.focus_start_s
        ):
            raise ValueError(f"{label} focused window end must be greater than its start")
        self.label = label
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    video_path: Path
    output_dir: Path
    broad_fps: float = 4.0
    focused_fps: float = 12.0
    focus_start_s: float | None = None
    focus_end_s: float | None = None
    source_offset_s: float | None = None
    model_name: str = DEFAULT_MODEL
    provider: str = "openai"
    mock: bool = False
    resume: bool = False
    max_duration_s: float = 120.0
    max_frames: int = 80
    max_individual_images: int = 12
    frames_per_sheet: int = 6
    sheet_columns: int = 3
    stage_images: str = "cited"
    # Run the blind pass at the same time as observer -> candidate. It needs only the frames,
    # so this saves one model call of wall time. Turn off to debug, or so the blind pass can
    # read the observer's images from a provider's prompt cache.
    parallel_passes: bool = True
    subjects: list[str] = []
    clips: list[ClipSpec] = []

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if self.broad_fps <= 0 or self.focused_fps <= 0:
            raise ValueError("Sampling rates must be positive")
        if (self.focus_start_s is None) ^ (self.focus_end_s is None):
            raise ValueError("Provide both --focus-start and --focus-end, or neither")
        if (
            self.focus_start_s is not None
            and self.focus_end_s is not None
            and self.focus_end_s <= self.focus_start_s
        ):
            raise ValueError("Focused window end must be greater than its start")
        if self.max_duration_s <= 0:
            raise ValueError("Max duration must be positive")
        if self.max_frames < 1 or self.max_individual_images < 1:
            raise ValueError("Frame limits must be at least 1")
        if self.frames_per_sheet < 1 or self.sheet_columns < 1:
            raise ValueError("Contact sheet layout must be positive")
        if self.provider not in {"openai", "anthropic", "muse"}:
            raise ValueError("Provider must be openai, anthropic, or muse")
        if self.stage_images not in {"cited", "all"}:
            raise ValueError("stage_images must be cited or all")
        labels = [clip.label for clip in self.clips]
        if len(labels) != len(set(labels)):
            raise ValueError("Clip labels must be unique")
        cleaned = [item.strip() for item in self.subjects if item.strip()]
        for item in cleaned:
            _reject_subject_steering(item)
        self.subjects = cleaned
        return self


_SUBJECT_STEERING = [
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
        r"\bhamstrings?\b",
        r"\bachilles\b",
        r"caleb williams",
        r"youtube\.com",
        r"publicly reported",
    )
]


def _reject_subject_steering(text: str) -> None:
    for pattern in _SUBJECT_STEERING:
        if pattern.search(text):
            raise ValueError(
                "A subject line may only identify who to watch. "
                f"Remove language matching '{pattern.pattern}'."
            )


def load_project_env() -> None:
    """Load the project .env file. Variables already set in the environment stay as they are."""
    seen: set[Path] = set()
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        load_env_file(path)


def load_env_file(path: Path) -> None:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key or not value:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value


def resolve_provider(explicit: str | None) -> str:
    if explicit:
        provider = explicit.strip().lower()
    elif os.environ.get("ANALYSIS_PROVIDER"):
        provider = os.environ["ANALYSIS_PROVIDER"].strip().lower()
    elif os.environ.get("MODEL_API_KEY") or os.environ.get("MUSE_API_KEY"):
        provider = "muse"
    elif os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        provider = "anthropic"
    else:
        provider = "openai"
    if provider not in {"openai", "anthropic", "muse"}:
        raise InjuryAnalysisError("Provider must be openai, anthropic, or muse")
    return provider


def resolve_model_name(explicit: str | None, provider: str) -> str:
    if explicit:
        return explicit
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if provider == "muse":
        return os.environ.get("MUSE_MODEL") or DEFAULT_MUSE_MODEL
    return os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL


def config_error(exc: Exception) -> InjuryAnalysisError:
    return InjuryAnalysisError(str(exc))
