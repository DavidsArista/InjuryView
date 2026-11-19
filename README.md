# Injury analysis

Evidence-first prototype for a short football clip. It inspects the video, extracts timestamped frames, and writes an observer timeline, mechanism candidates, a blind second look, a judge comparison, and a Markdown report.

This is not a medical-diagnosis tool. The report describes visible movement and possible mechanisms. Video alone cannot confirm a specific injury, tissue damage, or severity.

## What the first version does

```
video -> metadata -> timestamped frames -> contact sheets
      -> observer -> candidate analysis
      -> blind observer (fresh conversation, no prior report)
      -> judge -> report.md
```

Timestamps in the report come from frame presentation timestamps (`ffprobe`), not from the language model. If a model proposes different times, the pipeline replaces them before saving the artifacts.

The reference clip is a local file. This project does not download it.

Place it at:

```
data/raw/caleb_williams_injury.mp4
```

The analyzer is not told a public injury outcome, and it is not given a preferred mechanism for that clip.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` on `PATH`
- An Anthropic API key only if you want a real vision-model run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the pipeline on a synthetic clip

This checks the full artifact path without an API key:

```bash
python -m injury_analysis make-fixture --output data/fixtures/synthetic.mp4
python -m injury_analysis analyze \
  --video data/fixtures/synthetic.mp4 \
  --output runs/synthetic-mock \
  --mock
```

Mock output is labeled as a fixture. It is not a reading of the pixels.

## Run the reference clip

Broad sampling defaults to 4 frames per second. Focused sampling defaults to 12 frames per second and is used only when you pass a window. Choose that window after you watch the clip. Do not expect the tool to know it.

Mock rehearsal:

```bash
python -m injury_analysis analyze \
  --video data/raw/caleb_williams_injury.mp4 \
  --output runs/caleb-example \
  --broad-fps 4 \
  --focused-fps 12 \
  --focus-start START \
  --focus-end END \
  --mock
```

Vision-model run:

Put the Muse Spark key in `.env` in the project root (`MODEL_API_KEY=...`). That file is not committed. The shell environment is used instead when the variable is already set. The default model is `muse-spark-1.3`. Set `MUSE_MODEL` to change it. Use `--provider openai` with `OPENAI_API_KEY`, or `--provider anthropic` with `ANTHROPIC_API_KEY`, for the other adapters.

```bash
python -m injury_analysis analyze \
  --video data/raw/caleb_williams_injury.mp4 \
  --output runs/caleb-example \
  --broad-fps 4 \
  --focused-fps 12 \
  --focus-start START \
  --focus-end END
```

`--resume` reuses frames already extracted in the output directory only when the sampling settings match and each video has the same content hash (sha256) as when the frames were extracted. A different or re-exported video is refused, not silently reused. The saved observer report is reused only when its request fingerprint matches exactly: the same frames, prompt, subject lines, provider, and model. Otherwise the observer runs again. For example, mock output is never reused in a real run. The candidate, blind, and judge stages always run again. Runs made before fingerprints existed cannot be resumed.

`--stage-images cited` (the default) sends the candidate and judge stages only the frames that earlier passes cited: the observer's frames for the candidate, and the frames cited by either observer pass or the candidate report for the judge. Each stage's prompt lists only the frames it sees, and a citation of any other frame is rejected. When those frames do not fit as individual images, new contact sheets are built from them in `stage_sheets/`. Both observer passes always see every frame. `--stage-images all` sends every frame to every stage, for comparison. The report's limitations section says how many frames each stage saw.

The blind pass needs only the frames, so it runs at the same time as the observer and candidate stages. It still starts a fresh conversation and never receives the first report or the ranking. `--sequential` runs it after the candidate stage instead: slower, but useful when debugging, or when a provider's prompt cache could let the blind pass reuse the observer's images.

`--source-offset SECONDS` prints an upstream source time next to each clip-relative time. Analysis stages still see clip-relative times only.

Useful flags: `--broad-fps`, `--focused-fps`, `--sequential`, `--focus-start`, `--focus-end`, `--subject`, `--model`, `--mock`, `--resume`, `--max-duration`, `--max-frames`.

`--subject "QB, 18, Brown Jersey with White pants"` tells every pass who to watch. Repeat the flag to name other players so they are not confused with the first. The line is an identity, not an injury finding. A changed subject on `--resume` changes the observer's fingerprint, so the observer runs again.

## Artifacts

Each run writes:

```
runs/<run-id>/
  metadata.json
  frames/
  frame_manifest.json
  contact_sheets/
  stage_sheets/
  observer_report.json
  candidate_report.json
  blind_reanalysis.json
  judge_report.json
  report.md
  run_config.json
  invocations.jsonl
  usage.jsonl
  prompts/
```

`run_config.json` holds the latest invocation. `invocations.jsonl` adds one line per invocation with its status, which stages ran or were reused, and usage totals. `usage.jsonl` has one row per model call in the latest invocation: stage, attempt, image count, input and output tokens when the provider reports them, and latency. Each model artifact records an `input_fingerprint`, the hash of the request that produced it.

`prompts/` is an audit log of the text sent to each stage. The blind pass is a new conversation with the same frames and observer instructions. It does not receive the first report, the candidate ranking, or an external ground truth.

## Tests

```bash
pytest
```

The tests build short synthetic videos with ffmpeg. They check presentation timestamps (including files whose timestamps do not start at zero), resume fingerprints, multi-angle handling, schema validation, report sections, scoring, and that the blind pass cannot see the first report. They do not claim to analyze the reference clip.

An autouse fixture in `tests/conftest.py` keeps the tests away from real credentials. It stops the project `.env` from loading, removes provider variables, and blocks network connections. A key in `.env` cannot be used by `pytest`.

## Scoring against human labels

Scoring compares saved artifacts with labels a person wrote from the frames. It makes no model calls, and the analysis pipeline never opens the labels file. Agreement between the two observer passes is not treated as accuracy.

1. Get each video's hash. Labels identify videos by content, not by file name:

   ```bash
   python -m injury_analysis fingerprint --video data/raw/clip.mp4
   ```

2. Write labels in the shape of `eval/annotations.example.json`. Label only what the frames show, before reading any model output. Use `null` where the annotator cannot tell. Do not use public injury reports as labels.

3. Score one or more finished runs:

   ```bash
   python -m injury_analysis score \
     --annotations eval/annotations.json \
     --runs runs/clip-a runs/clip-b \
     --output eval/scores.json
   ```

For each angle and each observer pass, the scorer reports onset time error in seconds and frames, contact status, and the side of the first visible change. For each run it reports whether the leading candidate matches an optional mechanism label, how many judge claims cite frames outside their own frame list, and model calls, tokens, and latency from `usage.jsonl`. The onset is the earliest `onset` observation on that angle about the first `--subject`, or about a player marked `unclear`.

## Limits

- Local clips of one play (one or more camera angles), command line only.
- No player tracking, pose estimation, or training.
- The vision model sees individual frames when there are few of them, and labeled contact sheets when there are many. Original frames are kept either way.
- Clips longer than 120 seconds are rejected unless you raise `--max-duration`.
- A real reading of the reference clip requires the local file and `MODEL_API_KEY`. The mock path will not invent one.
