from __future__ import annotations

import base64
import copy

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.models.model_adapter import ModelRequest, UsageTracking


class AnthropicAdapter(UsageTracking):
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
        response = client.messages.create(
            model=self.model,
            max_tokens=12000,
            system=request.system,
            tools=[
                {
                    "name": "submit_analysis",
                    "description": "Submit the structured analysis for this stage.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "submit_analysis"},
            messages=[{"role": "user", "content": _content(request)}],
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise InjuryAnalysisError(
                f"{request.stage} response was truncated. Lower the frame count or try again."
            )
        payload = _tool_payload(response)
        return request.response_model.model_validate(payload)

    def _client(self):
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise InjuryAnalysisError(
                "ANTHROPIC_API_KEY is not set. Export it, or re-run with --mock to use the fixture model."
            )
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client


def inline_schema(schema: dict) -> dict:
    """Inline local $ref entries so the tool schema is self-contained."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name not in defs:
                    raise InjuryAnalysisError(f"Schema reference {node['$ref']} could not be resolved")
                resolved = resolve(copy.deepcopy(defs[name]))
                extra = {key: resolve(value) for key, value in node.items() if key != "$ref"}
                if isinstance(resolved, dict):
                    merged = dict(resolved)
                    merged.update(extra)
                    return merged
                return resolved
            return {
                key: resolve(value)
                for key, value in node.items()
                if key not in {"$defs", "$schema"}
            }
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def _content(request: ModelRequest) -> list[dict]:
    blocks: list[dict] = []
    for image in request.images:
        blocks.append({"type": "text", "text": image.label})
        data = base64.standard_b64encode(image.path.read_bytes()).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _media_type(image.path.suffix),
                    "data": data,
                },
            }
        )
    blocks.append({"type": "text", "text": request.user_text})
    return blocks


def _media_type(suffix: str) -> str:
    lowered = suffix.lower()
    if lowered in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if lowered == ".png":
        return "image/png"
    raise InjuryAnalysisError(f"Unsupported image type for the vision model: {suffix}")


def _tool_payload(response) -> dict:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_analysis":
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    raise InjuryAnalysisError("The model response did not include a submit_analysis tool call")
