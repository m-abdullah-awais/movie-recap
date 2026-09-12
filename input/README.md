# input

Put the movie you want to analyse in this folder.

Any container ffmpeg can read works, such as `.mkv`, `.mp4`, `.avi`, `.mov`,
`.m4v`, `.ts`, or `.webm`. Nothing keys off the file extension, so a mislabelled
file is still read correctly.

With exactly one movie in here, no path is needed anywhere:

```powershell
.\Run.bat
.\.venv\Scripts\python.exe analyze.py all
```

If you keep several movies in here, the tool will list them and ask you to name
one, because guessing which film you meant would waste minutes of encoding.

Subtitles are picked up automatically from inside the movie file. If a film has
no usable subtitle track, you can drop a matching `.srt` next to it and it will
be used instead of speech recognition. The stem has to match, so `movie.mkv`
pairs with `movie.srt` or `movie.en.srt`.

Nothing in this folder is ever modified. The original file is only read, and the
analysis writes to `cache/` instead.

The contents of this folder are not tracked by git. This file is the exception,
so that the folder still exists after a fresh clone.
