@echo off
setlocal EnableDelayedExpansion
title Local Movie Recap Generator - Analysis Runner

rem ==========================================================
rem  Guided test runner for analysis stages 1 to 3.
rem
rem  Usage:
rem    Run.bat                         walk through every step
rem    Run.bat "D:\films\movie.mkv"    same, movie supplied up front
rem    Run.bat "D:\films\movie.mkv" -y run without asking
rem
rem  Every location this touches is inside the project folder.
rem  The source movie is only ever read, never modified.
rem ==========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

rem Project-scoped caches. These match setup.ps1 so that running the pipeline
rem from this script can never install or cache anything outside the project.
set "UV_PYTHON_INSTALL_DIR=%ROOT%\.python"
set "UV_PROJECT_ENVIRONMENT=%ROOT%\.venv"
set "UV_CACHE_DIR=%ROOT%\.uv-cache"
set "PIP_CACHE_DIR=%ROOT%\.uv-cache\pip"
set "XDG_CACHE_HOME=%ROOT%\.uv-cache"
set "HF_HOME=%ROOT%\.models"
set "UV_HTTP_TIMEOUT=300"
set "UV_CONCURRENT_DOWNLOADS=2"

set "PY=%ROOT%\.venv\Scripts\python.exe"
set "AUTO=0"
set "MOVIE="
set "FAILED=0"

rem ---- read arguments -------------------------------------------------------
rem Each branch jumps to its own label. Writing this as
rem   if cond set X=1 & shift
rem would be wrong, because cmd.exe parses that as "(if cond set X=1) & shift"
rem and the shift would run whether the condition matched or not.
:parse_args
if "%~1"=="" goto parsed
if /i "%~1"=="-y" goto arg_auto
if /i "%~1"=="/y" goto arg_auto
set "MOVIE=%~1"
shift
goto parse_args

:arg_auto
set "AUTO=1"
shift
goto parse_args

:parsed

cls
echo ==============================================================
echo    LOCAL MOVIE RECAP GENERATOR
echo    Guided test run for analysis stages 1 to 3
echo ==============================================================
echo.
echo  What this pipeline does:
echo.
echo    Stage 1  ingest     reads the film's dialogue, from embedded
echo                        subtitles, a sidecar file, or by listening
echo    Stage 2  proxy      builds a 480p analysis copy plus 16 kHz
echo                        mono audio in a single pass
echo    Stage 3  scenemap   turns scene changes into a shot list
echo.
echo  This script runs each check in order and asks first.
echo  Press Enter to accept a step, or type N to skip it.
echo.
if "%AUTO%"=="1" echo  Running in automatic mode, no questions will be asked.
if "%AUTO%"=="1" echo.
echo  Project folder: %ROOT%
echo.
if "%AUTO%"=="1" goto step0
pause

rem ==========================================================
rem  STEP 0  prerequisites
rem ==========================================================
:step0
call :header "STEP 0 of 6   Prerequisites"
echo  Checking that the tools the pipeline needs are present.
echo.

where ffmpeg >nul 2>&1
if errorlevel 1 goto no_ffmpeg
echo   [ok]      ffmpeg found
goto check_ffprobe
:no_ffmpeg
echo   [MISSING] ffmpeg is not on PATH
set "FAILED=1"

:check_ffprobe
where ffprobe >nul 2>&1
if errorlevel 1 goto no_ffprobe
echo   [ok]      ffprobe found
goto check_venv
:no_ffprobe
echo   [MISSING] ffprobe is not on PATH
set "FAILED=1"

:check_venv
if not exist "%PY%" goto need_setup
echo   [ok]      project virtual environment found
goto step0_done

:need_setup
echo   [MISSING] the project virtual environment does not exist yet
echo.
echo  Setup installs Python 3.11 and the dependencies inside this
echo  folder. It downloads about 80 MB and can take several minutes
echo  on a slow connection.
echo.
call :confirm "Run setup now"
if errorlevel 1 goto setup_declined
echo.
where uv >nul 2>&1
if errorlevel 1 goto no_uv
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup.ps1"
if errorlevel 1 goto setup_failed
echo.
echo   Setup finished.
goto step0_done

:no_uv
echo   [MISSING] uv is not on PATH, so setup cannot run.
echo             Install uv, then run this script again.
goto abort

:setup_failed
echo.
echo   Setup did not complete. Re-running it resumes from the cache,
echo   so try this script again. Downloads already finished are kept.
goto abort

:setup_declined
echo.
echo   Cannot continue without the virtual environment.
goto abort

:step0_done
if "%FAILED%"=="1" goto missing_tools
echo.
echo   All prerequisites present.
goto step1

:missing_tools
echo.
echo   ffmpeg and ffprobe are required. Put them on PATH, or set the
echo   FFMPEG and FFPROBE environment variables to their full paths.
goto abort

rem ==========================================================
rem  STEP 1  doctor
rem ==========================================================
:step1
call :header "STEP 1 of 6   Environment check"
echo  Confirms the interpreter, reports whether Intel Quick Sync is
echo  usable for fast proxy encoding, and verifies that every cache
echo  location resolves inside this folder.
echo.
call :confirm "Run the environment check"
if errorlevel 1 goto step2
echo.
"%PY%" analyze.py doctor
if errorlevel 1 echo.
if errorlevel 1 echo   The check reported a problem. Read the FAIL and MISSING lines above.
goto step2

rem ==========================================================
rem  STEP 2  smoke test
rem ==========================================================
:step2
call :header "STEP 2 of 6   Smoke test, no movie required"
if not exist "%ROOT%\temp\synth_subs.mkv" goto no_samples
if not exist "%ROOT%\temp\static.mp4" goto no_samples
echo  Runs all three stages against two tiny generated clips.
echo.
echo    synth_subs.mkv   has embedded subtitles and 3 scene cuts
echo    static.mp4       is one motionless shot with no cuts, which
echo                     exercises the degraded fallback path
echo.
echo  This proves the pipeline works before spending minutes on a
echo  real film.
echo.
call :confirm "Run the smoke test"
if errorlevel 1 goto step3
echo.
echo   ---- clip 1 of 2, expect an embedded subtitle source and 3 shots
echo.
"%PY%" analyze.py all "temp\synth_subs.mkv"
echo.
echo   ---- clip 2 of 2, expect DEGRADED uniform_fallback, which is correct
echo.
"%PY%" analyze.py all "temp\static.mp4"
goto step3

:no_samples
echo  The generated sample clips are not present, so this step is
echo  skipped. It is optional.
echo.
goto step3

rem ==========================================================
rem  STEP 3  inspect a real film
rem ==========================================================
:step3
call :header "STEP 3 of 6   Inspect a real film"
echo  Probes the file and reports its streams, which subtitle track
echo  would be chosen, and whether speech recognition is needed.
echo  This writes nothing and takes about a second.
echo.
echo  Any container ffmpeg can read works, such as mp4, mkv, avi,
echo  or mov.
echo.
if not "%MOVIE%"=="" goto have_movie

rem Prefer the drop-in folder. Counting the candidates here means a single
rem movie in input needs no path at all, while several are reported rather
rem than guessed between.
set "FOUND="
set "COUNT=0"
if not exist "%ROOT%\input" goto ask_movie
for %%F in ("%ROOT%\input\*.mkv" "%ROOT%\input\*.mp4" "%ROOT%\input\*.m4v" "%ROOT%\input\*.avi" "%ROOT%\input\*.mov" "%ROOT%\input\*.webm" "%ROOT%\input\*.wmv" "%ROOT%\input\*.flv" "%ROOT%\input\*.mpg" "%ROOT%\input\*.mpeg" "%ROOT%\input\*.m2ts" "%ROOT%\input\*.ts") do call :note_candidate "%%~fF"

if "%COUNT%"=="0" goto no_input
if "%COUNT%"=="1" goto one_input
echo   %COUNT% movies are in the input folder, so it is not clear which
echo   one you mean. Give the path, or leave only one file in there.
echo.
goto ask_movie

:one_input
set "MOVIE=%FOUND%"
echo   Found one movie in the input folder:
echo     %MOVIE%
echo.
call :confirm "Use this one"
if errorlevel 1 goto ask_movie
goto have_movie

:no_input
echo   The input folder is empty.
echo.
echo   Put the movie in:  %ROOT%\input
echo   then run this script again, and no path will be needed.
echo.

:ask_movie
set "MOVIE="
set /p "MOVIE=Full path to the movie, or press Enter to stop here: "
if "!MOVIE!"=="" goto finish_early
set "MOVIE=!MOVIE:"=!"

:have_movie
if not exist "!MOVIE!" goto bad_movie
echo.
echo   Using: !MOVIE!
echo.
"%PY%" analyze.py info "!MOVIE!"
if errorlevel 1 goto probe_failed
goto step4

:bad_movie
echo.
echo   That path does not point to a file.
echo.
goto ask_movie

:probe_failed
echo.
echo   ffmpeg could not read that file. It may be corrupt or may not
echo   be a media file.
goto abort

rem ==========================================================
rem  STEP 4  the real measurement
rem ==========================================================
:step4
call :header "STEP 4 of 6   Full analysis, the real measurement"
echo  Runs all three stages on the film and prints per-stage timings.
echo  This is the number that decides whether the remaining stages
echo  are feasible as designed.
echo.
echo  Measured on a 94 minute HEVC film on this machine: 7 minutes
echo  14 seconds, which is 13 times realtime. Scale that by your
echo  film's runtime. Software decode is roughly 25 percent slower.
echo  Nearly all of it is the one unavoidable read of the film.
echo  Watch the percentage during the proxy stage.
echo.
echo  If the film has no usable subtitles, a 250 MB speech model is
echo  downloaded once into the .models folder.
echo.
call :confirm "Run the full analysis"
if errorlevel 1 goto finish
echo.
"%PY%" analyze.py all "!MOVIE!"
if errorlevel 1 goto analysis_failed
goto step5

:analysis_failed
echo.
echo   The analysis did not finish. The error above explains why.
echo   Nothing was written to the movie file itself.
goto abort

rem ==========================================================
rem  STEP 5  prove the cache works
rem ==========================================================
:step5
call :header "STEP 5 of 6   Prove the cache works"
echo  Re-runs the exact same command. Every stage should report
echo  'hit' and the whole run should finish in about a second,
echo  because each stage is keyed by a content hash.
echo.
echo  Then it retunes the scene threshold. Only scenemap should
echo  recompute. The proxy must stay cached, which is what makes
echo  tuning cheap instead of a multi-minute re-encode.
echo.
call :confirm "Run the cache checks"
if errorlevel 1 goto step6
echo.
echo   ---- re-run, expect three cache hits
echo.
"%PY%" analyze.py all "!MOVIE!"
echo.
echo   ---- retune the threshold, expect proxy=hit and scenemap=computed
echo.
"%PY%" analyze.py all "!MOVIE!" --force-stage scenemap --threshold 12
goto step6

rem ==========================================================
rem  STEP 6  review the output
rem ==========================================================
:step6
call :header "STEP 6 of 6   Review the output"
echo  What to check by eye:
echo.
echo    proxy.mp4        should play and look like the film
echo    scenes.json      a few shot boundaries should land on real cuts
echo    transcript.json  cue times should match the spoken dialogue
echo    timings.json     per-stage timings for every run so far
echo.
echo  Two numbers worth your judgement:
echo.
echo    shot count       roughly 1500 to 2500 is normal for a feature.
echo                     Far more means the threshold is too low, so
echo                     raise it with --threshold and re-run step 5.
echo    coverage_ratio   around 0.3 to 0.5 for a dialogue-heavy film.
echo                     Much lower usually means the wrong subtitle
echo                     track was picked.
echo.
rem Automatic mode never opens a window, since it may be running unattended.
if "%AUTO%"=="1" goto finish
call :confirm "Open the cache folder now"
if errorlevel 1 goto finish
if exist "%ROOT%\cache" start "" explorer "%ROOT%\cache"
if not exist "%ROOT%\cache" echo   No cache folder exists yet.
goto finish

rem ==========================================================
rem  endings
rem ==========================================================
:finish_early
echo.
echo   Stopped before running a real film. The smoke test results
echo   above still tell you whether the pipeline is working.
goto done

:finish
call :header "Finished"
echo  Artifacts are in the cache folder, one subfolder per film.
echo.
echo  Useful commands from here:
echo.
echo    .\.venv\Scripts\python.exe analyze.py cache-list
echo    .\.venv\Scripts\python.exe analyze.py all "MOVIE" --threshold 12
echo    .\.venv\Scripts\python.exe analyze.py proxy "MOVIE" --no-qsv
echo    .\.venv\Scripts\python.exe analyze.py --help
echo.
goto done

:abort
echo.
echo ==============================================================
echo    Stopped. Nothing outside this project folder was changed.
echo ==============================================================
echo.
if not "%AUTO%"=="1" pause
endlocal
exit /b 1

:done
echo.
rem Do not block on a keypress in automatic mode, which may be unattended.
if not "%AUTO%"=="1" pause
endlocal
exit /b 0

rem ==========================================================
rem  helpers
rem ==========================================================
:header
echo.
echo --------------------------------------------------------------
echo  %~1
echo --------------------------------------------------------------
exit /b 0

:note_candidate
rem Called once per matching file. A for loop over a pattern that matches
rem nothing still yields nothing, so COUNT staying at zero means an empty
rem folder.
set /a COUNT+=1
set "FOUND=%~1"
exit /b 0

:confirm
if "%AUTO%"=="1" exit /b 0
set "REPLY="
set /p "REPLY=%~1? [Y/n] "
if /i "!REPLY!"=="n" exit /b 1
if /i "!REPLY!"=="no" exit /b 1
exit /b 0
