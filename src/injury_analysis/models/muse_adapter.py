from __future__ import annotations

import base64
import json

from injury_analysis.constants import MUSE_BASE_URL
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.anthropic_adapter import inline_schema
from injury_analysis.models.model_adapter import ModelRequest
from injury_analysis.models.openai_adapter import _media_type, chat_usage


class MuseAdapter:
    """Muse Spark through the Meta Model API. Each call is a new conversation.

    Muse Spark rejects a forced tool choice, so the structured result comes back
    as JSON constrained by response_format.
    """

    def __init__(self, model: str, api_key: str | None = None, client=None):
        self.model = model
        self.api_key = api_key
        self.client = client
        self.last_usage: dict | None = None

    def complete(self, request: ModelRequest):
        self.last_usage = None
        client = self._client()
        schema = inline_schema(request.response_model.model_json_schema())
        response = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=16000,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": _content(request)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis",
                    "schema": schema,
                },
            },
        )
        self.last_usage = chat_usage(response)
        return request.response_model.model_validate(_json_payload(response, request.stage))

    def _client(self):
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise InjuryAnalysisError(
                "MODEL_API_KEY is not set. Add the Muse Spark key to .env in the project root, "
                "or re-run with --mock to use the fixture model."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=MUSE_BASE_URL)
        return self.client


def _content(request: ModelRequest) -> list[dict]:
    parts: list[dict] = []
    for image in request.images:
        parts.append({"type": "text", "text": image.label})
        data = base64.standard_b64encode(image.path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{_media_type(image.path.suffix)};base64,{data}"},
            }
        )
    parts.append({"type": "text", "text": request.user_text})
    return parts


def _load_json_object(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    found = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return found if isinstance(found, dict) else None
    return None


def _json_payload(response, stage: str) -> dict:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise InjuryAnalysisError(f"{stage} response did not include a completion choice")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise InjuryAnalysisError(
            f"{stage} response was truncated. Lower the frame count or try again."
        )
    content = getattr(choice.message, "content", None)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(getattr(part, "text", ""))
            for part in content
        )
    if not isinstance(content, str) or not content.strip():
        finish = getattr(choice, "finish_reason", None)
        raise InjuryAnalysisError(
            f"{stage} response did not include JSON (finish_reason={finish})"
        )
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    payload = _load_json_object(text)
    if payload is None:
        preview = " ".join(text.split())[:240]
        raise InjuryAnalysisError(f"{stage} response was not valid JSON. Response began: {preview}")
    if not isinstance(payload, dict):
        raise InjuryAnalysisError(f"{stage} response JSON was not an object")
    return payload
