from __future__ import annotations

import base64
import json

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.anthropic_adapter import inline_schema
from injury_analysis.models.model_adapter import ModelRequest, UsageTracking


class OpenAIAdapter(UsageTracking):
    """Vision-model adapter. Each call is a new conversation."""

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
            max_completion_tokens=12000,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": _content(request)},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_analysis",
                        "description": "Submit the structured analysis for this stage.",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "submit_analysis"}},
        )
        self.last_usage = chat_usage(response)
        return request.response_model.model_validate(_tool_payload(response, request.stage))

    def _client(self):
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise InjuryAnalysisError(
                "OPENAI_API_KEY is not set. Add it to .env in the project root, "
                "or re-run with --mock to use the fixture model."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)
        return self.client


def chat_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def _content(request: ModelRequest) -> list[dict]:
    parts: list[dict] = []
    for image in request.images:
        parts.append({"type": "text", "text": image.label})
        data = base64.standard_b64encode(image.path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_media_type(image.path.suffix)};base64,{data}",
                    "detail": "high",
                },
            }
        )
    parts.append({"type": "text", "text": request.user_text})
    return parts


def _media_type(suffix: str) -> str:
    lowered = suffix.lower()
    if lowered in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if lowered == ".png":
        return "image/png"
    raise InjuryAnalysisError(f"Unsupported image type for the vision model: {suffix}")


def _tool_payload(response, stage: str) -> dict:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise InjuryAnalysisError(f"{stage} response did not include a completion choice")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise InjuryAnalysisError(
            f"{stage} response was truncated. Lower the frame count or try again."
        )
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None or getattr(function, "name", None) != "submit_analysis":
            continue
        raw = function.arguments
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InjuryAnalysisError(f"{stage} tool call was not valid JSON") from exc
        elif isinstance(raw, dict):
            payload = raw
        else:
            raise InjuryAnalysisError(f"{stage} tool call had an unexpected payload")
        if isinstance(payload, dict):
            return payload
    raise InjuryAnalysisError("The model response did not include a submit_analysis tool call")
