from __future__ import annotations

import hashlib
import json
from pathlib import Path

from injury_analysis.models.model_adapter import ModelRequest


def file_sha256(path: Path) -> str:
    """Content hash of a file. Identifies a video even when it is renamed or re-exported to the same path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_fingerprint(request: ModelRequest, producer: str, model_name: str) -> str:
    """Hash of everything that determines a stage's output: prompt text, image bytes, schema, and model.

    A saved artifact may be reused only when this matches exactly.
    """
    payload = {
        "stage": request.stage,
        "system": request.system,
        "user_text": request.user_text,
        "images": [
            [image.relative_path, file_sha256(image.path), image.label] for image in request.images
        ],
        "schema": request.response_model.model_json_schema(),
        "producer": producer,
        "model_name": model_name,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
