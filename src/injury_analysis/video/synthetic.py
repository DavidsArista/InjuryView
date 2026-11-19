from __future__ import annotations

from pathlib import Path

from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.video.ffmpeg import require_ffmpeg, run_command


def write_synthetic_video(
    path: Path,
    *,
    duration_s: float = 2.0,
    fps: str = "30",
    size: str = "320x240",
) -> None:
    """Write a small generated clip so the pipeline can be exercised without broadcast footage."""
    if duration_s <= 0 or duration_s > 30:
        raise InjuryAnalysisError("Synthetic fixture duration must be between 0 and 30 seconds")
    require_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={fps}:duration={duration_s}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
