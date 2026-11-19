DISCLAIMER = (
    "This report describes visible movement and possible mechanisms. "
    "It is not a medical diagnosis. Video alone cannot confirm a specific injury, "
    "tissue damage, or severity."
)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_MUSE_MODEL = "muse-spark-1.3"
MUSE_BASE_URL = "https://api.meta.ai/v1"
TIMESTAMP_AUTHORITY = "video_presentation_timestamp"

STAGE_OBSERVER = "observer"
STAGE_CANDIDATE = "candidate"
STAGE_BLIND = "blind_reanalysis"
STAGE_JUDGE = "judge"

OBSERVER_BLOCK = "observer_report_json"
CANDIDATE_BLOCK = "candidate_report_json"
BLIND_BLOCK = "blind_reanalysis_json"
