from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from injury_analysis.config import (
    ClipSpec,
    RunConfig,
    config_error,
    load_project_env,
    resolve_model_name,
    resolve_provider,
)
from injury_analysis.constants import DISCLAIMER
from injury_analysis.errors import InjuryAnalysisError
from injury_analysis.evaluation import load_annotations, score_runs, summarize, write_scores
from injury_analysis.pipeline import run_pipeline
from injury_analysis.provenance import file_sha256
from injury_analysis.video.synthetic import write_synthetic_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="injury_analysis",
        description=DISCLAIMER,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one or more local video clips")
    analyze.add_argument(
        "--video",
        required=True,
        action="append",
        type=Path,
        help="Local video file. Repeat once per camera angle.",
    )
    analyze.add_argument(
        "--label",
        action="append",
        default=None,
        help="Angle name for the matching --video, such as wide-sideline. Required when more than one video is passed.",
    )
    analyze.add_argument(
        "--clip-window",
        action="append",
        default=None,
        help="Focused window for the matching --video, as START-END in that file's own seconds.",
    )
    analyze.add_argument("--output", required=True, type=Path, help="Run directory for artifacts")
    analyze.add_argument("--broad-fps", type=float, default=4.0, help="Frames per second across the clip")
    analyze.add_argument(
        "--focused-fps",
        type=float,
        default=12.0,
        help="Frames per second inside the focused window",
    )
    analyze.add_argument("--focus-start", type=float, default=None, help="Focused window start, clip seconds")
    analyze.add_argument("--focus-end", type=float, default=None, help="Focused window end, clip seconds")
    analyze.add_argument(
        "--source-offset",
        type=float,
        default=None,
        help="Optional seconds to add when displaying an upstream source time",
    )
    analyze.add_argument(
        "--provider",
        choices=("openai", "anthropic", "muse"),
        default=None,
        help="Vision provider. Defaults from ANALYSIS_PROVIDER, or muse when MODEL_API_KEY is set",
    )
    analyze.add_argument("--model", default=None, help="Model name for the selected provider")
    analyze.add_argument("--mock", action="store_true", help="Use the deterministic fixture model")
    analyze.add_argument("--resume", action="store_true", help="Reuse extracted frames in the output directory")
    analyze.add_argument("--max-duration", type=float, default=120.0, help="Reject clips longer than this many seconds")
    analyze.add_argument("--max-frames", type=int, default=80, help="Maximum extracted frames")
    analyze.add_argument(
        "--max-individual-images",
        type=int,
        default=12,
        help="Send individual frames at or below this count; otherwise send contact sheets",
    )
    analyze.add_argument("--frames-per-sheet", type=int, default=6)
    analyze.add_argument(
        "--stage-images",
        choices=("cited", "all"),
        default="cited",
        help=(
            "Frames sent to the candidate and judge stages. 'cited' sends only frames cited by the "
            "earlier passes; 'all' sends every extracted frame. Both observer passes always see every frame."
        ),
    )
    analyze.add_argument("--sheet-columns", type=int, default=3)
    analyze.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Run the blind pass after the candidate stage instead of alongside the observer. "
            "Slower; use it to debug, or so the blind pass can hit a provider's prompt cache."
        ),
    )
    analyze.add_argument(
        "--subject",
        action="append",
        default=None,
        help=(
            "Who to follow, for example: 'QB, 18, Brown Jersey with White pants'. "
            "The first line is the player to watch. Repeat the flag to name other players "
            "so they are not confused with that player. This is not an injury finding."
        ),
    )

    score = subparsers.add_parser(
        "score",
        help="Score finished run directories against human labels. Makes no model calls.",
    )
    score.add_argument("--annotations", required=True, type=Path, help="Labels file, such as eval/annotations.json")
    score.add_argument("--runs", required=True, nargs="+", type=Path, help="Run directories to score")
    score.add_argument("--output", type=Path, default=None, help="Optional JSON file for every scored row")

    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="Print the sha256 that labels use to identify a video",
    )
    fingerprint.add_argument("--video", required=True, action="append", type=Path)

    fixture = subparsers.add_parser("make-fixture", help="Write a synthetic MP4 for a local dry run")
    fixture.add_argument("--output", required=True, type=Path)
    fixture.add_argument("--duration", type=float, default=2.0)
    fixture.add_argument("--fps", default="30")
    fixture.add_argument("--size", default="320x240")
    return parser


def _clips_from_args(args) -> tuple[Path, list[ClipSpec]]:
    videos = list(args.video)
    labels = list(args.label or [])
    windows = list(args.clip_window or [])
    if len(videos) == 1 and not labels and not windows:
        return videos[0], []
    if args.focus_start is not None or args.focus_end is not None:
        raise InjuryAnalysisError(
            "Use --clip-window START-END for each --video when passing more than one angle."
        )
    if len(labels) != len(videos):
        raise InjuryAnalysisError("Pass one --label for each --video.")
    if len(windows) != len(videos):
        raise InjuryAnalysisError("Pass one --clip-window START-END for each --video.")
    clips = []
    for path, label, window in zip(videos, labels, windows, strict=True):
        start_text, separator, end_text = window.partition("-")
        if not separator:
            raise InjuryAnalysisError(f"Clip window {window!r} must look like 5.2-8.8")
        try:
            start_s = float(start_text)
            end_s = float(end_text)
        except ValueError as exc:
            raise InjuryAnalysisError(f"Clip window {window!r} must look like 5.2-8.8") from exc
        clips.append(
            ClipSpec(video_path=path, label=label, focus_start_s=start_s, focus_end_s=end_s)
        )
    return videos[0], clips


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "score":
            return _score(args)
        if args.command == "fingerprint":
            for path in args.video:
                if not path.is_file():
                    raise InjuryAnalysisError(f"Video file does not exist: {path}")
                print(f"{file_sha256(path)}  {path}")
            return 0
    except InjuryAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    load_project_env()
    try:
        if args.command == "make-fixture":
            write_synthetic_video(
                args.output,
                duration_s=args.duration,
                fps=args.fps,
                size=args.size,
            )
            print(f"Wrote {args.output}")
            return 0
        provider = resolve_provider(args.provider)
        video_path, clips = _clips_from_args(args)
        config = RunConfig(
            video_path=video_path,
            clips=clips,
            output_dir=args.output,
            broad_fps=args.broad_fps,
            focused_fps=args.focused_fps,
            focus_start_s=args.focus_start,
            focus_end_s=args.focus_end,
            source_offset_s=args.source_offset,
            provider=provider,
            model_name=resolve_model_name(args.model, provider),
            mock=args.mock,
            resume=args.resume,
            max_duration_s=args.max_duration,
            max_frames=args.max_frames,
            max_individual_images=args.max_individual_images,
            frames_per_sheet=args.frames_per_sheet,
            sheet_columns=args.sheet_columns,
            stage_images=args.stage_images,
            parallel_passes=not args.sequential,
            subjects=args.subject or [],
        )
        result = run_pipeline(config)
    except ValidationError as exc:
        print(f"error: {config_error(exc)}", file=sys.stderr)
        return 2
    except InjuryAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.report_path)
    return 0


def _score(args) -> int:
    annotations = load_annotations(args.annotations)
    angle_rows, run_rows = score_runs(list(args.runs), annotations)
    if args.output is not None:
        write_scores(args.output, angle_rows, run_rows)
    print(summarize(angle_rows, run_rows), end="")
    return 0

