from __future__ import annotations

import shutil
import subprocess

from injury_analysis.errors import InjuryAnalysisError


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        joined = " and ".join(missing)
        raise InjuryAnalysisError(
            f"{joined} must be installed and available on PATH. "
            "On macOS, install them with: brew install ffmpeg"
        )


def run_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise InjuryAnalysisError(
            f"Required executable '{args[0]}' was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        tail = detail[-2000:]
        preview = " ".join(args[:8])
        raise InjuryAnalysisError(f"Command failed: {preview}\n{tail}") from exc
    return completed.stdout
