from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.video.frames import FrameManifest


@dataclass(frozen=True)
class ImagePayload:
    relative_path: str
    path: Path
    label: str


@dataclass(frozen=True)
class ModelRequest:
    stage: str
    system: str
    user_text: str
    images: list[ImagePayload]
    response_model: type[BaseModel]
    manifest: FrameManifest


class ModelAdapter(Protocol):
    def complete(self, request: ModelRequest) -> BaseModel:
        """Return one schema-valid model object for a fresh request."""


class UsageTracking:
    """Gives an adapter a `last_usage` that is private to the calling thread.

    The observer and blind passes run at the same time on one adapter. A plain attribute
    would let one call's token counts overwrite the other's before UsageRecorder reads them.
    """

    @property
    def last_usage(self) -> dict | None:
        local = self.__dict__.get("_usage_local")
        return getattr(local, "value", None) if local is not None else None

    @last_usage.setter
    def last_usage(self, value: dict | None) -> None:
        local = self.__dict__.setdefault("_usage_local", threading.local())
        local.value = value


@dataclass
class UsageRecorder:
    """Wraps an adapter and records one row per model call: tokens, image count, and latency.

    Adapters that report tokens expose them as `last_usage`. The mock does not, so its rows
    have no token counts.
    """

    adapter: ModelAdapter
    records: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def complete(self, request: ModelRequest) -> BaseModel:
        with self._lock:
            attempt = 1 + sum(1 for record in self.records if record["stage"] == request.stage)
        started = time.perf_counter()
        error = None
        try:
            return self.adapter.complete(request)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            usage = getattr(self.adapter, "last_usage", None) or {}
            with self._lock:
                self.records.append(
                    {
                        "stage": request.stage,
                        "attempt": attempt,
                        "images": len(request.images),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "latency_s": round(time.perf_counter() - started, 3),
                        "error": error,
                    }
                )


def complete_with_retry(
    adapter: ModelAdapter,
    request: ModelRequest,
    validate,
    normalize=None,
) -> tuple[BaseModel, list[ModelRequest]]:
    """Call the model, then retry once if schema or evidence checks fail.

    `normalize` runs before `validate` and fixes structural rules in code (for example, a
    required candidate that the model left out), so those never cost a second model call.
    Rules that need judgement stay in `validate` and still trigger the retry.
    """
    attempts: list[ModelRequest] = []
    pending = request
    last_errors = ["The model was not called"]
    for _ in range(2):
        attempts.append(pending)
        try:
            report = adapter.complete(pending)
        except ValidationError as exc:
            last_errors = [f"Response did not match the schema: {exc}"[:4000]]
        except InjuryAnalysisError as exc:
            last_errors = [str(exc)[:4000]]
        else:
            if normalize is not None:
                report = normalize(report)
            last_errors = list(validate(report))
            if not last_errors:
                return report, attempts
        pending = _with_feedback(request, last_errors)
    raise InjuryAnalysisError(
        "Model output failed validation after one retry: " + "; ".join(last_errors)
    )


def _with_feedback(request: ModelRequest, errors: list[str]) -> ModelRequest:
    feedback = (
        "Your previous submission was rejected. Fix every issue and submit again.\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\nCite only frame IDs from the manifest. "
        "Do not state a diagnosis, a grade, a recovery time, or an established tissue injury."
    )
    return ModelRequest(
        stage=request.stage,
        system=request.system,
        user_text=request.user_text + "\n\n" + feedback,
        images=request.images,
        response_model=request.response_model,
        manifest=request.manifest,
    )


def render_prompt_log(request: ModelRequest) -> str:
    image_lines = [f"- {image.relative_path} | {image.label}" for image in request.images]
    images = "\n".join(image_lines) if image_lines else "(none)"
    return (
        f"STAGE: {request.stage}\n\n"
        f"SYSTEM:\n{request.system}\n\n"
        f"USER:\n{request.user_text}\n\n"
        f"IMAGES:\n{images}\n"
    )
