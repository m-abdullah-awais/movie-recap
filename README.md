# Local Movie Recap Generator

A terminal tool that turns a full feature film into a finished, narrated recap
video with no manual editing. Drop in a movie file and get back a 10 to 20 minute
video in which an AI-written narration walks through the plot, spoken by a
synthetic voice, over clips pulled automatically from the film that match what the
narrator is describing at that moment.

Everything runs locally on a CPU-only machine. There are no paid APIs, no cloud
services, no web interface, no Docker, and no database.

## Design principle

The AI reads the movie, it never watches it. Story understanding comes entirely
from the subtitle and dialogue text. Computer vision is used only for retrieval,
to find which shot matches a given line of narration.

The narration script is written first and footage is fitted to it afterwards. This
is what makes synchronisation automatic: the spoken duration of each narration
line defines exactly how much video must sit behind it, so no forced aligner is
needed.

## Pipeline

| # | Stage | What it does | Status |
| --- | --- | --- | --- |
| 1 | `ingest` | Extract dialogue from embedded subtitles, a sidecar file, or speech recognition | Built |
| 2 | `proxy` | One ffmpeg read of the film producing scene data and a 16 kHz mono wav | Built |
| 3 | `scenemap` | Coarse shot boundaries across the whole film | Built |
| 4 | `story` | Claude reads the dialogue and outputs cast, acts, and plot beats | Planned |
| 5 | `script` | Claude turns the story into narration segments with visual queries | Planned |
| 6 | `index` | Shot detection and CLIP embedding, restricted to relevant regions | Planned |
| 7 | `narrate` | Piper text to speech per segment, cached by text hash | Planned |
| 8 | `select` | Score and pick shots for each narration segment | Planned |
| 9 | `render` | ffmpeg assembles the final video, stream-copying the source | Planned |

Stages 1 to 3 are complete and are the subject of this release. The remaining
stages are deliberately unwritten until real timings from a full length film
confirm the approach is viable on the target hardware.

The film is read exactly once, in stage 2. No full-film video proxy is written,
because nothing downstream needs one: stage 6 indexes only the regions the
narration script actually references, and stage 9 stream-copies the original.
Pass `--with-proxy` to write a 480p copy anyway, which is useful for seeing what
the pipeline saw.

## Measured performance

On an Intel i7-8650U, 4 cores at 2.1 GHz, with Intel UHD 620 graphics, against a
94 minute HEVC Main 10 1080p film carrying English subtitles.

| Stage | Time |
| --- | --- |
| `ingest` | 0:01 |
| `proxy` | 7:13 |
| `scenemap` | 0:00 |
| Total | 7:14, or 13x realtime |

Reading the film is effectively the entire cost, and it is unavoidable. Scale by
your film's runtime: a 2 hour film lands near 9 minutes. Software decoding is
roughly 25 percent slower. Writing a 480p proxy as well adds about 6 minutes,
which is why it is off by default.

Two approaches were measured and rejected. Keeping frames on the GPU through
`vpp_qsv` would avoid downloading them, but fails on this hardware with a Direct3D
texture allocation error. Skipping B-frames to cut decode work leaves only 9 of
every 720 frames on this film, far too coarse for shot detection.

## Requirements

- Windows with PowerShell
- `ffmpeg` and `ffprobe` on `PATH`, or `FFMPEG` and `FFPROBE` pointing at them
- `uv` on `PATH`
- Roughly 200 MB of disk for the toolchain, plus about 200 MB per film analysed

Python 3.11 is required and is installed by the setup script. The system Python is
not used, because `ctranslate2` publishes no wheels for Python 3.14.

## Installation

Everything is installed inside the project directory. Nothing is installed
globally or to the user profile.

```powershell
.\setup.ps1
```

The script redirects every tool cache into the project, fetches CPython 3.11 into
`.python`, creates `.venv`, installs the dependencies, and then verifies
containment by running the `doctor` command.

The dependency download is around 80 MB. The script raises the HTTP timeout to
300 seconds and limits itself to two parallel downloads, because the default 30
second timeout combined with eight simultaneous transfers causes every one of
them to time out on a slow link. On a poor connection the first run can take
several minutes, and re-running the script resumes from the cache rather than
starting over.

## Quick start

`Run.bat` walks through every check in order and asks before each step. It runs
setup itself if the virtual environment is missing.

```powershell
.\Run.bat
```

It can also take the film up front, and run without asking.

```powershell
.\Run.bat "D:\films\movie.mkv"
.\Run.bat "D:\films\movie.mkv" -y
```

The six steps are prerequisites, environment check, a smoke test on two tiny
generated clips that needs no film, a probe of your film, the full analysis with
timings, and a cache verification that proves re-runs are free.

## Usage

The runner is a convenience. Every command it issues can be run directly.

```powershell
# Check the environment and confirm nothing leaks outside the project
.\.venv\Scripts\python.exe analyze.py doctor

# Inspect a file without doing any work
.\.venv\Scripts\python.exe analyze.py info "D:\films\movie.mkv"

# Run stages 1 to 3 and print per-stage timings
.\.venv\Scripts\python.exe analyze.py all "D:\films\movie.mkv"
```

Any container ffmpeg can read is accepted, including `.mp4`, `.mkv`, `.avi`, and
`.mov`. Nothing in the tool keys off the file extension.

### Individual stages

Each stage runs on its own for debugging.

```powershell
.\.venv\Scripts\python.exe analyze.py ingest   "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py proxy    "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py scenemap "D:\films\movie.mkv"
```

### Useful flags

| Flag | Effect |
| --- | --- |
| `--force` | Ignore the cache and recompute every stage |
| `--force-stage scenemap` | Recompute one stage and leave the others cached |
| `--with-proxy` | Also write a full-film 480p proxy, for inspection only |
| `--proxy-height 360` | Height of that proxy when `--with-proxy` is used |
| `--threshold 6` | Lower the scene change threshold to detect more cuts |
| `--no-qsv` | Disable Quick Sync and decode entirely in software |
| `--json` | Emit machine readable output instead of a table |
| `--quiet` | Suppress progress output |

### Cache management

```powershell
.\.venv\Scripts\python.exe analyze.py cache-list
.\.venv\Scripts\python.exe analyze.py cache-clear "D:\films\movie.mkv"
```

## How it works

### Caching

Every stage reads and writes artifacts keyed by a content hash, so re-running a
stage is skippable by default. This matters because the later stages are
expensive, and iterating on them would be unaffordable if each attempt re-ran a
multi-minute proxy encode.

A film is identified by a fingerprint over its size plus three 8 MB samples taken
at the start, middle, and end. Hashing a multi-gigabyte file in full would cost
minutes, while this runs in under a second and still survives renames and copies.

Each stage key also covers that stage's own parameters, so lowering the scene
threshold invalidates `scenemap` without disturbing the proxy. Artifacts are
written to a temporary name and moved into place only on success, so an
interrupted run never leaves a truncated file that a later run would trust.

### Soft degradation

No stage crashes where it can degrade instead.

- No usable subtitles falls back to speech recognition
- Bitmap subtitles are detected and skipped, because they carry images not text
- A subtitle track covering too little of the runtime is rejected and the next
  candidate is tried, which catches mislabelled forced and commentary tracks
- Quick Sync failure falls back to software decoding
- Missing scene data falls back to a uniform shot grid
- A speech model that cannot be downloaded is reported clearly, and the stages
  that do not need dialogue still complete

### Hardware acceleration

Stage 2 decodes through Intel Quick Sync when it is available, falling back to
software automatically. Quick Sync is a fixed-function media engine rather than
GPU compute, and no AI work touches it.

The Quick Sync decoder is named explicitly per codec rather than requested
through `-hwaccel`. The `-hwaccel` form needs an output pixel format declared up
front, and the obvious choice of `nv12` is 8 bit only, so a 10 bit source such as
HEVC Main 10 fails to initialise and silently falls back to software. Naming the
decoder lets it choose its own format, and the filter graph normalises the bit
depth afterwards.

Because support varies by codec and inputs are arbitrary, the working command is
still discovered per file by running a short real pass rather than by trusting
the encoder list.

### Scene detection

Detection runs during the single read, on a hard downscale to 160 pixels wide,
which makes it almost free. Frame differences survive that aggressive a
downsample. Chroma is kept rather than converting to grayscale, so two scenes of
similar brightness but different colour still separate.

Detection uses a permissive floor and records every candidate cut's score.
Stage 3 then applies your threshold to those scores. That is what makes
`--threshold` cheap: retuning it re-runs stage 3 in under a second instead of
re-reading the film.

## Cache artifacts

Written to `cache/<source_id>/`.

| File | Stage | Contents |
| --- | --- | --- |
| `transcript.json` | ingest | Dialogue cues, chosen source, coverage statistics |
| `dialogue.srt` | ingest | Normalised subtitles as extracted |
| `proxy.mp4` | proxy | 480p copy, only when `--with-proxy` is used |
| `proxy.wav` | proxy | 16 kHz mono audio for speech recognition |
| `scdet.raw.txt` | proxy | Candidate scene changes with their scores |
| `scenes.json` | scenemap | Shot list with statistics |
| `timings.json` | all | Per-stage timings for every run |

## Project layout

```
Run.bat                 guided test runner
analyze.py              entry point
setup.ps1               project-scoped bootstrap
pyproject.toml          dependencies, pinned to Python 3.11
src/recap/
  cli.py                Typer command line interface
  config.py             paths, tunables, environment containment
  cache.py              content hash cache layer
  ffmpeg.py             ffmpeg and ffprobe wrappers
  probe.py              stream selection and encoder probing
  srt.py                subtitle parsing and normalisation
  timing.py             timing report
  stages/               ingest, proxy, scenemap
```

## Troubleshooting

**`doctor` reports the interpreter is not inside the project.** You are running
the system Python. Use `.\.venv\Scripts\python.exe analyze.py` instead.

**Setup fails with a network timeout.** Re-run `setup.ps1`. Completed downloads
are cached in `.uv-cache` and are not fetched again.

**ffmpeg or ffprobe not found.** Put them on `PATH`, or set the `FFMPEG` and
`FFPROBE` environment variables to their full paths.

**Scene detection finds far too many or too few cuts.** Retune with
`--threshold`. Detection during the proxy pass runs at a permissive floor and
records each candidate cut's score, so the threshold is applied afterwards in
stage 3. Changing it costs seconds rather than a full re-encode.

**A transcript looks nearly empty.** Run `info` on the file to see which
subtitle tracks exist and which one would be chosen. A low coverage ratio usually
means only a forced or commentary track is present, in which case speech
recognition is the better source.

## Developer

**Muhammad Abdullah Awais**
Full Stack Developer

| Channel | Link |
| --- | --- |
| Website | [www.abdullahawais.com](https://www.abdullahawais.com) |
| Email | [contact@abdullahawais.com](mailto:contact@abdullahawais.com) |
| LinkedIn | [m-abdullah-awais-programmer](https://www.linkedin.com/in/m-abdullah-awais-programmer) |
| GitHub | [m-abdullah-awais](https://github.com/m-abdullah-awais) |
| YouTube | [@m_abdullah_awais](https://www.youtube.com/@m_abdullah_awais) |
| Instagram | [@m_abdullah_awais](https://www.instagram.com/m_abdullah_awais) |
