import pytest

from injury_analysis.video.metadata import inspect_video


def test_probe_reads_duration_resolution_rate_and_time_base(synthetic_video):
    metadata, frames = inspect_video(synthetic_video)

    assert metadata.duration_s == pytest.approx(2.0, abs=0.05)
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.frame_rate == pytest.approx(30.0)
    assert metadata.time_base
    assert metadata.codec_name
    assert metadata.probed_frame_count == len(frames)
    assert frames[0].time_s == pytest.approx(0.0, abs=1e-6)
    assert frames[-1].time_s > 1.5
    assert all(right.time_s >= left.time_s for left, right in zip(frames, frames[1:]))


def test_rejects_a_clip_over_the_duration_limit(synthetic_video):
    from injury_analysis.errors import InjuryAnalysisError

    with pytest.raises(InjuryAnalysisError, match="exceeds the limit"):
        inspect_video(synthetic_video, max_duration_s=0.5)
