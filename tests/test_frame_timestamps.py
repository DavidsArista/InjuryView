import pytest

from injury_analysis.stages.materials import model_images
from injury_analysis.video.contact_sheet import build_contact_sheets
from injury_analysis.video.frames import extract_frames, select_frames, target_times
from injury_analysis.video.metadata import ProbedFrame, inspect_video
from injury_analysis.video.synthetic import write_synthetic_video


def test_target_times_follow_the_requested_rate():
    times = target_times(0.0, 2.0, 4.0)

    assert times[0] == 0.0
    assert times[-1] == pytest.approx(2.0)
    gaps = [right - left for left, right in zip(times, times[1:])]
    assert gaps
    assert all(gap == pytest.approx(0.25) for gap in gaps)


def test_focused_sampling_adds_frames_and_can_share_broad_frames():
    probed = [
        ProbedFrame(index=index, pts_s=index / 30, time_s=index / 30, width=320, height=240)
        for index in range(60)
    ]
    broad = select_frames(
        probed,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=None,
        focus_end_s=None,
        max_frames=80,
    )
    both = select_frames(
        probed,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.5,
        focus_end_s=1.0,
        max_frames=80,
    )

    assert len(both) > len(broad)
    assert [frame.time_s for frame in both] == sorted(frame.time_s for frame in both)
    assert "both" in {frame.sampling for frame in both}
    assert "focused" in {frame.sampling for frame in both}


def test_extracted_timestamps_match_presentation_timestamps(synthetic_video, tmp_path):
    metadata, probed = inspect_video(synthetic_video)
    manifest = extract_frames(
        synthetic_video,
        tmp_path,
        metadata,
        probed,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=0.4,
        focus_end_s=1.2,
        max_frames=80,
        source_offset_s=10,
    )
    before_sheets = sorted((tmp_path / "frames").glob("*.jpg"))
    manifest = build_contact_sheets(manifest, tmp_path, columns=3, frames_per_sheet=6)
    after_sheets = sorted((tmp_path / "frames").glob("*.jpg"))

    by_index = {frame.index: frame for frame in probed}
    assert manifest.frames
    assert [frame.frame_id for frame in manifest.frames] == [
        f"f{number:03d}" for number in range(1, len(manifest.frames) + 1)
    ]
    assert after_sheets == before_sheets
    assert manifest.contact_sheets
    for record, image in zip(manifest.frames, after_sheets, strict=True):
        source = by_index[record.source_frame_index]
        assert image.name == f"{record.frame_id}.jpg"
        assert record.time_s == pytest.approx(source.time_s, abs=1e-6)
        assert record.pts_s == pytest.approx(source.pts_s, abs=1e-6)
        assert record.source_time_s == pytest.approx(record.time_s + 10, abs=1e-6)
        assert image.is_file() and image.stat().st_size > 0
    for sheet in manifest.contact_sheets:
        sheet_path = tmp_path / sheet.path
        assert sheet_path.is_file() and sheet_path.stat().st_size > 0
        assert all(frame_id in {frame.frame_id for frame in manifest.frames} for frame_id in sheet.frame_ids)

    sheet_images = model_images(manifest, tmp_path, max_individual=1)
    assert sheet_images
    assert all(image.relative_path.startswith("contact_sheets/") for image in sheet_images)


def test_timestamps_are_not_synthesized_from_a_nominal_30fps_index(tmp_path):
    path = tmp_path / "ntsc.mp4"
    write_synthetic_video(path, duration_s=1, fps="30000/1001", size="160x120")
    metadata, probed = inspect_video(path)
    manifest = extract_frames(
        path,
        tmp_path / "run",
        metadata,
        probed,
        broad_fps=4,
        focused_fps=12,
        focus_start_s=None,
        focus_end_s=None,
        max_frames=80,
        source_offset_s=None,
    )
    by_index = {frame.index: frame for frame in probed}
    deviations = []
    for record in manifest.frames:
        assert record.time_s == pytest.approx(by_index[record.source_frame_index].time_s, abs=1e-6)
        deviations.append(abs(record.time_s - (record.source_frame_index / 30)))
    assert max(deviations) > 1e-4
