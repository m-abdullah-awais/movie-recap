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
| 4 | `story` | Claude reads the dialogue and outputs cast, acts, and plot beats | Built |
| 5 | `script` | Claude turns the story into narration segments with visual queries | Built |
| 6 | `index` | Keyframes and CLIP embeddings, restricted to relevant regions | Built |
| 7 | `narrate` | Piper text to speech per segment, cached by text hash | Built |
| 8 | `select` | Score and pick shots for each narration segment | Built |
| 9 | `render` | ffmpeg assembles the final video from the original | Built |

All nine stages are complete and measured on a real film.

The film is read exactly once, in stage 2. No full-film video proxy is written,
because nothing downstream needs one: stage 6 indexes only the regions the
narration script actually references, and stage 9 stream-copies the original.
Pass `--with-proxy` to write a 480p copy anyway, which is useful for seeing what
the pipeline saw.

## Measured performance

On an Intel i7-8650U, 4 cores at 2.1 GHz, with Intel UHD 620 graphics, against a
94 minute HEVC Main 10 1080p film carrying English subtitles.

| Stage | Time | Cost |
| --- | --- | --- |
| `ingest` | 0:01 | |
| `proxy` | 7:13 | |
| `scenemap` | 0:00 | |
| `story` | 3:44 | $1.58 |
| `script` | 2:33 | $0.46 |
| `index` | 2:03 | |
| `narrate` | 1:14 | |
| `select` | 0:00 | |
| `render` | 7:12 | |
| Total | 24:00 | $2.04 |

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

Put a film in the `input` folder and run `Run.bat`. It lists what it finds,
asks which one, and then runs all nine stages to completion with no further
questions. The finished recap appears in `output`.

```powershell
.\Run.bat
.\Run.bat 2
```

Passing a number starts straight on that film in the list. Each stage prints
when it begins and how long it took, so it is always clear what is done and
what is left. Setup runs automatically if the project has not been set up yet.

Expect roughly 25 minutes for a 90 minute film on modest hardware. Anything
already computed is reused, so a repeat run is far quicker.

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
.\.venv\Scripts\python.exe analyze.py story    "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py script   "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py index    "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py narrate  "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py select   "D:\films\movie.mkv"
.\.venv\Scripts\python.exe analyze.py render   "D:\films\movie.mkv"
```

### Useful flags

| Flag | Effect |
| --- | --- |
| `--force` | Ignore the cache and recompute every stage |
| `--force-stage scenemap` | Recompute one stage and leave the others cached |
| `--with-proxy` | Also write a full-film 480p proxy, for inspection only |
| `--refine` | Refine shot boundaries with PySceneDetect. Much slower |
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

### Story understanding

Stage 4 sends one Claude call per ten minute window of dialogue, then a single
synthesis call merges the windows into a whole-film structure. Every call is
cached individually by the hash of its prompt, so a failed run never repays for
the windows that already succeeded, and a re-run costs nothing.

Calls use `--system-prompt`, which replaces the default system prompt outright
and keeps this project's own memory files out of an analysis call. Flags stay
identical between calls so each one after the first reuses the same prompt cache
prefix. That is the difference between five cents and twenty-five cents a call.

### Writing the script

Stage 5 writes one Claude call per act, in order rather than in parallel, because
each act's narration has to follow on from the last without repeating it. The
word budget is shared out by how much of the film each act covers.

Spoiler control is computed in Python, not asked of the model. Each segment
carries a ceiling on how late in the film its footage may come from, set by the
next twist the narration has not reached yet and capped a fixed distance ahead of
the current position. Without the cap, an early segment could be backed by a shot
from the finale.

The script is written before any footage is chosen. That is what makes
synchronisation automatic: once a line has been spoken and measured, its duration
says exactly how much video must sit behind it.

### Narrowing before indexing

The script anchors every narration segment to a moment in the film, so only the
stretches around those anchors can ever supply footage. Indexing the whole film
instead is the easiest way to lose the time budget.

The window around each anchor is adaptive, not fixed. A fixed window narrows
nothing when the narration is dense: on a 94 minute film with 77 segments a
median 62 seconds apart, a window of plus or minus 90 seconds merged into four
regions covering 99 percent of the film. The window now halves until coverage
meets a target fraction, which gave 77 regions over 15 percent of that film.

Keyframes are named by their timestamp rather than by position. A positional
name is reused by a later run whose shot boundaries differ, which silently pairs
a shot with a frame from somewhere else.

Refinement with PySceneDetect is available but off by default. Measured on the
same film it cost 6 minutes 54 seconds and found 24 extra shots out of 243,
because the regions are short and the existing boundaries already average about
three seconds.

### Speaking, then choosing footage

Narration is spoken before any footage is chosen, and each line is measured. That
measured duration is what tells the next stage how much video to pack behind the
line, which is why no forced aligner is needed anywhere.

Each line is cached by the hash of its own text and the speaking rate, so editing
one segment re-speaks only that segment and two identical lines are synthesised
once.

Speaking rate needed calibrating against measurement rather than assumption. The
Lessac medium voice runs at about 205 words per minute at its default, which is
rushed. The response to the length scale is not linear: 1.6 gives roughly 170,
and 2.0 drops to 134. The default of 1.6 measured 154.8 words per minute on a
real script.

Footage for each line spans the spoken seconds plus the silence that follows it.
Covering only the spoken part would leave the video one gap per line shorter than
the audio, which across 77 lines is about 27 seconds of narration cut off the end.

Shots are scored on similarity to the line's visual query, closeness to the
moment being described, how comfortably their length fits the clip band, and
penalties for reuse and darkness. The spoiler ceiling is a hard exclusion rather
than a penalty, and a shot may be used only a limited number of times.

### Matching shots to meaning

With the CLIP encoders present, each shot's keyframe and each line's visual query
are embedded into the same 512 dimensional space, and a dot product gives the
similarity. Cosine scores sit in a narrow positive band, so they are rescaled per
line before weighting, otherwise time proximity would dominate simply because it
already spans zero to one.

Measured on a real film, 84 percent of chosen clips score above 0.5 after
rescaling and 31 percent above 0.8. The useful sign is that matching overrides
chronology when it should: a line calling for a boy running from police at night
was given a shot 50 seconds away from its narration anchor, because that shot
actually showed it.

Preprocessing is verified rather than assumed. Wrong channel order or wrong
normalisation constants raise no error, they just quietly degrade every score, so
the check compares the darkest and brightest keyframes against matching prompts
and confirms each prefers its own.

Without the encoders the stage falls back to time proximity, which still produces
a finished video, with footage that follows the plot chronologically rather than
matching each line.

### Rendering

One ffmpeg invocation reads every clip straight out of the original film through
the concat demuxer, with an in point and an out point per clip, so no
intermediate files are written. The narration sits over the film's own audio,
ducked underneath it by a sidechain compressor keyed on the narration itself.

Stream copying the video is not the default, despite being much faster. A copy
can only begin on a keyframe, but these clip boundaries come from shot detection
and narration timing and fall wherever they fall. Copying would shift every clip
to an earlier keyframe or emit corrupt leading frames. `--copy-video` is there if
speed matters more than exact cuts.

The audio track is chosen by ordinal among audio streams, not by original stream
index, because the concat demuxer renumbers them. On a dual-audio film this is
the difference between the intended language and the wrong one.

Subtitles are timed to the spoken duration, not to the footage allotted to the
line. The footage also covers the silence that follows, so using it would hold
each caption on screen through the gap and butt it against the next.

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
| `story.json` | story | Cast, acts, beats, setups and payoffs, twists |
| `story_calls/` | story | One cached response per Claude call |
| `script.json` | script | Narration segments with visual queries and spoiler ceilings |
| `script_calls/` | script | One cached response per Claude call |
| `shots.json` | index | Narrowed regions, shots, keyframes, brightness |
| `keyframes/` | index | One 224 by 224 frame per shot, named by timestamp |
| `clip_index.npy` | index | Shot embeddings, when CLIP is available |
| `query_index.npy` | index | Visual query embeddings, when CLIP is available |
| `narration.json` | narrate | Spoken lines with measured durations and offsets |
| `audio/` | narrate | One wav per distinct line, named by text hash |
| `edl.json` | select | Edit decision list, clips chosen per line |
| `final.mp4` | render | The finished recap video |
| `subtitle.srt` | render | Narration subtitles, sidecar rather than burned in |
| `timings.json` | all | Per-stage timings for every run |

## Project layout

```
Run.bat                 pick a film, then run everything
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
