@echo off
setlocal EnableDelayedExpansion
title Local Movie Recap Generator - Cleanup

rem ==========================================================
rem  Deletes reclaimable files, one category at a time.
rem
rem  Usage:
rem    Cleanup.bat        pick from a menu
rem    Cleanup.bat 1      run one option directly
rem    Cleanup.bat 1 -y   run it without confirming
rem
rem  Options are ordered by what it costs to get the data back.
rem  Nothing here ever touches the input folder or the git history.
rem ==========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

set "AUTO=0"
set "PICK="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="-y" goto arg_auto
if /i "%~1"=="/y" goto arg_auto
set "PICK=%~1"
shift
goto parse
:arg_auto
set "AUTO=1"
shift
goto parse
:parsed

if not "%PICK%"=="" goto dispatch

:menu
cls
echo ==============================================================
echo    CLEANUP
echo ==============================================================
echo.
call :sizes
echo.
echo  Safe, cheap to rebuild:
echo.
echo    1  Scratch files in temp            seconds to nothing
echo    2  Python bytecode caches           rebuilt automatically
echo    3  Keyframes and narration audio    about 3 minutes
echo.
echo  Reclaims a lot, costs real time to rebuild:
echo.
echo    4  The 16 kHz audio track           needs a full film read, 7 min
echo    5  Rendered videos in the cache     re-render, about 8 min
echo    6  The uv download cache            80 MB re-download, slow link
echo.
echo  Think before choosing:
echo.
echo    7  Cached AI responses              costs about 2 dollars to redo
echo    8  Downloaded models                CLIP cannot be re-downloaded here
echo    9  The whole toolchain              full setup again, very slow
echo.
echo    A  Everything in groups 1 to 3      the safe sweep
echo    B  Whole cache folder for all films keeps models and toolchain
echo    C  Everything from finished recaps    cache, videos and temp
echo.
echo    0  Exit without deleting anything
echo.
set "PICK="
set /p "PICK=Choose an option: "
if "%PICK%"=="" goto menu

:dispatch
if /i "%PICK%"=="0" goto done
if /i "%PICK%"=="1" goto opt_temp
if /i "%PICK%"=="2" goto opt_pycache
if /i "%PICK%"=="3" goto opt_derived
if /i "%PICK%"=="4" goto opt_wav
if /i "%PICK%"=="5" goto opt_video
if /i "%PICK%"=="6" goto opt_uvcache
if /i "%PICK%"=="7" goto opt_calls
if /i "%PICK%"=="8" goto opt_models
if /i "%PICK%"=="9" goto opt_toolchain
if /i "%PICK%"=="A" goto opt_safe
if /i "%PICK%"=="B" goto opt_cache
if /i "%PICK%"=="C" goto opt_finished
echo.
echo   Not an option: %PICK%
echo.
if "%AUTO%"=="1" goto done
pause
goto menu

rem ==========================================================
rem  1  scratch
rem ==========================================================
:opt_temp
call :header "Scratch files in temp"
echo  Working files left behind by the tooling. The two sample clips
echo  used by Run.bat's smoke test are kept.
echo.
echo  NOTE: if you put the CLIP model zip in temp, it is in here too.
echo  Those model files are already extracted into .models, so the zip
echo  is no longer needed, but it is your only local copy while
echo  Hugging Face is rate limiting this address.
echo.
call :confirm "Delete temp scratch files"
if errorlevel 1 goto after
for %%F in ("temp\*.txt" "temp\*.json" "temp\*.log" "temp\*.srt") do if exist "%%~F" del /f /q "%%~F" >nul 2>&1
for /d %%D in ("temp\qsvtest" "temp\lean" "temp\testcache*" "temp\pace" "temp\laddertest") do if exist "%%~D" rd /s /q "%%~D" >nul 2>&1
echo   Done. Sample clips and any zip were left in place.
goto after

rem ==========================================================
rem  2  bytecode
rem ==========================================================
:opt_pycache
call :header "Python bytecode caches"
echo  Regenerated automatically the next time anything runs.
echo.
call :confirm "Delete bytecode caches"
if errorlevel 1 goto after
for /f "delims=" %%D in ('dir /s /b /ad __pycache__ 2^>nul ^| findstr /v /i "\.venv"') do rd /s /q "%%D" >nul 2>&1
echo   Done.
goto after

rem ==========================================================
rem  3  derived analysis data
rem ==========================================================
:opt_derived
call :header "Keyframes and narration audio"
echo  Keyframes are re-extracted in about 2 minutes, narration is
echo  re-spoken in about 1 minute. The film is not read again.
echo.
echo  The JSON artifacts are kept, so the story and the script survive
echo  and cost nothing to reuse.
echo.
call :confirm "Delete keyframes and narration audio"
if errorlevel 1 goto after
for /d %%D in ("cache\*") do (
    if exist "%%~D\keyframes" rd /s /q "%%~D\keyframes" >nul 2>&1
    if exist "%%~D\audio" rd /s /q "%%~D\audio" >nul 2>&1
    if exist "%%~D\narration_track.wav" del /f /q "%%~D\narration_track.wav" >nul 2>&1
    if exist "%%~D\gap.wav" del /f /q "%%~D\gap.wav" >nul 2>&1
    if exist "%%~D\index.meta.json" del /f /q "%%~D\index.meta.json" >nul 2>&1
    if exist "%%~D\narrate.meta.json" del /f /q "%%~D\narrate.meta.json" >nul 2>&1
)
echo   Done.
goto after

rem ==========================================================
rem  4  proxy audio
rem ==========================================================
:opt_wav
call :header "The 16 kHz audio track"
echo  About 170 MB per film. Only needed for speech recognition, so a
echo  film with subtitles never uses it.
echo.
echo  Rebuilding it means reading the whole film again, about 7 minutes
echo  for a 90 minute film, because it is produced during that one pass.
echo.
call :confirm "Delete the extracted audio tracks"
if errorlevel 1 goto after
for /d %%D in ("cache\*") do (
    if exist "%%~D\proxy.wav" del /f /q "%%~D\proxy.wav" >nul 2>&1
    if exist "%%~D\proxy.meta.json" del /f /q "%%~D\proxy.meta.json" >nul 2>&1
)
echo   Done.
goto after

rem ==========================================================
rem  5  rendered videos in the cache
rem ==========================================================
:opt_video
call :header "Rendered videos in the cache"
echo  About 330 MB per film. Re-rendering takes about 8 minutes.
echo.
echo  Anything already published to the output folder is NOT touched
echo  and stays playable, because it is a separate directory entry to
echo  the same data.
echo.
call :confirm "Delete rendered videos from the cache"
if errorlevel 1 goto after
for /d %%D in ("cache\*") do (
    if exist "%%~D\final.mp4" del /f /q "%%~D\final.mp4" >nul 2>&1
    if exist "%%~D\render.meta.json" del /f /q "%%~D\render.meta.json" >nul 2>&1
)
echo   Done. The output folder was left alone.
goto after

rem ==========================================================
rem  6  uv cache
rem ==========================================================
:opt_uvcache
call :header "The uv download cache"
echo  About 415 MB of downloaded packages. The virtual environment
echo  keeps working, so nothing breaks today.
echo.
echo  But if the environment is ever rebuilt, those 80 MB of wheels get
echo  downloaded again. On this connection that took 53 minutes.
echo.
call :confirm "Delete the uv download cache"
if errorlevel 1 goto after
if exist ".uv-cache" rd /s /q ".uv-cache" >nul 2>&1
echo   Done.
goto after

rem ==========================================================
rem  7  cached AI responses
rem ==========================================================
:opt_calls
call :header "Cached AI responses"
echo  Only a few hundred kilobytes, so this reclaims almost nothing.
echo.
echo  It is on the list because the opposite matters: these are the
echo  cached story and script calls. Deleting them means paying for
echo  them again, roughly 2 dollars and 6 minutes per film.
echo.
echo  Delete these only if you want the story rewritten from scratch.
echo.
call :confirm "Delete cached AI responses"
if errorlevel 1 goto after
for /d %%D in ("cache\*") do (
    if exist "%%~D\story_calls" rd /s /q "%%~D\story_calls" >nul 2>&1
    if exist "%%~D\script_calls" rd /s /q "%%~D\script_calls" >nul 2>&1
    if exist "%%~D\story.meta.json" del /f /q "%%~D\story.meta.json" >nul 2>&1
    if exist "%%~D\script.meta.json" del /f /q "%%~D\script.meta.json" >nul 2>&1
)
echo   Done.
goto after

rem ==========================================================
rem  8  models
rem ==========================================================
:opt_models
call :header "Downloaded models"
echo  About 490 MB: the CLIP encoders and the Kokoro voice model.
echo.
echo  READ THIS FIRST. Hugging Face is rate limiting this address, so
echo  the CLIP encoders cannot currently be downloaded again. They came
echo  from a zip placed in temp by hand. If that zip is gone too, shot
echo  matching drops back to timing only until the limit clears.
echo.
echo  The Kokoro model is not downloaded by this tool either. It was
echo  copied in by hand, so deleting it leaves the robotic system voice
echo  until you put it back.
echo.
call :confirm "Delete downloaded models anyway"
if errorlevel 1 goto after
if exist ".models" rd /s /q ".models" >nul 2>&1
echo   Done. Run: scripts\analyze.py fetch-models  to try getting them back.
goto after

rem ==========================================================
rem  9  toolchain
rem ==========================================================
:opt_toolchain
call :header "The whole toolchain"
echo  About 500 MB: the virtual environment, the bundled Python, and
echo  anything in .tools, which is where setup puts the programs this
echo  computer did not already have, such as ffmpeg and the AI engine.
echo.
echo  Nothing in this project will run afterwards until Setup.bat has
echo  been run again. That downloads up to 550 MB, which took the best
echo  part of an hour on this connection.
echo.
call :confirm "Delete the virtual environment, Python and .tools"
if errorlevel 1 goto after
if exist ".venv" rd /s /q ".venv" >nul 2>&1
if exist ".python" rd /s /q ".python" >nul 2>&1
if exist ".tools" rd /s /q ".tools" >nul 2>&1
echo   Done. Run Setup.bat before using the tool again.
goto after

rem ==========================================================
rem  A  safe sweep
rem ==========================================================
:opt_safe
call :header "The safe sweep, groups 1 to 3"
echo  Scratch files, bytecode caches, keyframes and narration audio.
echo  Everything here rebuilds in about 3 minutes and none of it costs
echo  money or needs a download.
echo.
call :confirm "Run the safe sweep"
if errorlevel 1 goto after
for %%F in ("temp\*.txt" "temp\*.json" "temp\*.log" "temp\*.srt") do if exist "%%~F" del /f /q "%%~F" >nul 2>&1
for /d %%D in ("temp\qsvtest" "temp\lean" "temp\testcache*" "temp\pace" "temp\laddertest") do if exist "%%~D" rd /s /q "%%~D" >nul 2>&1
for /f "delims=" %%D in ('dir /s /b /ad __pycache__ 2^>nul ^| findstr /v /i "\.venv"') do rd /s /q "%%D" >nul 2>&1
for /d %%D in ("cache\*") do (
    if exist "%%~D\keyframes" rd /s /q "%%~D\keyframes" >nul 2>&1
    if exist "%%~D\audio" rd /s /q "%%~D\audio" >nul 2>&1
    if exist "%%~D\narration_track.wav" del /f /q "%%~D\narration_track.wav" >nul 2>&1
    if exist "%%~D\gap.wav" del /f /q "%%~D\gap.wav" >nul 2>&1
    if exist "%%~D\index.meta.json" del /f /q "%%~D\index.meta.json" >nul 2>&1
    if exist "%%~D\narrate.meta.json" del /f /q "%%~D\narrate.meta.json" >nul 2>&1
)
echo   Done.
goto after

rem ==========================================================
rem  B  whole cache
rem ==========================================================
:opt_finished
call :header "Everything from finished recaps"
echo  Clears the work for films that have been through the pipeline:
echo.
echo    cache            every analysis artifact, for every film
echo    output           the rendered videos and their subtitle files
echo    output samples   the generated voice samples
echo    temp             scratch files, keeping the two sample clips
echo.
echo  Kept: the input films, the downloaded models, and the Python
echo  toolchain.
echo.
echo  THIS DELETES YOUR FINISHED RECAPS. They are the point of the
echo  whole exercise, so move anything worth keeping out of the
echo  output folder first.
echo.
echo  Rebuilding from nothing takes about 25 minutes for a 90 minute
echo  film and costs roughly 2 dollars in AI calls, because the
echo  cached story and script go too.
echo.
call :confirm "Delete all finished recaps and their cache"
if errorlevel 1 goto after
call :confirm "This removes the videos in output. Really delete them"
if errorlevel 1 goto after

if exist "cache" rd /s /q "cache" >nul 2>&1

rem Flattened rather than nested inside a parenthesised if block. cmd.exe
rem mis-parses a for loop placed inside one, which fails the whole option.
for %%F in ("output\*.mp4" "output\*.srt" "output\*.mkv") do if exist "%%~F" del /f /q "%%~F" >nul 2>&1
if exist "output\voice-samples" rd /s /q "output\voice-samples" >nul 2>&1

for %%F in ("temp\*.txt" "temp\*.json" "temp\*.log" "temp\*.srt" "temp\*.wav") do if exist "%%~F" del /f /q "%%~F" >nul 2>&1
for /d %%D in ("temp\qsvtest" "temp\lean" "temp\testcache*" "temp\pace" "temp\kpace") do if exist "%%~D" rd /s /q "%%~D" >nul 2>&1
for /f "delims=" %%D in ('dir /s /b /ad __pycache__ 2^>nul ^| findstr /v /i "\.venv"') do rd /s /q "%%D" >nul 2>&1

echo   Done. Input films, models and the toolchain were left alone.
goto after

rem ==========================================================
rem  B  whole cache
rem ==========================================================
:opt_cache
call :header "The whole cache folder"
echo  Every analysis artifact for every film, currently about 600 MB.
echo.
echo  This includes the cached AI responses, so the story and the
echo  script would be paid for again, roughly 2 dollars per film. A
echo  full re-run from scratch takes about 24 minutes for a 90 minute
echo  film.
echo.
echo  Models, the toolchain, the input folder and anything already in
echo  the output folder are all left alone.
echo.
call :confirm "Delete the entire cache folder"
if errorlevel 1 goto after
if exist "cache" rd /s /q "cache" >nul 2>&1
echo   Done.
goto after

rem ==========================================================
rem  endings and helpers
rem ==========================================================
:after
echo.
call :sizes
echo.
if "%AUTO%"=="1" goto done
set "PICK="
set /p "PICK=Another option, or 0 to finish: "
if "%PICK%"=="" goto done
goto dispatch

:done
echo.
echo  Nothing outside this project folder was touched.
if not "%AUTO%"=="1" pause
endlocal
exit /b 0

:header
echo.
echo --------------------------------------------------------------
echo  %~1
echo --------------------------------------------------------------
exit /b 0

:confirm
rem Deleting is not undoable, so the default here is no. An explicit Y is
rem required, unlike Run.bat where the default is yes.
if "%AUTO%"=="1" exit /b 0
set "REPLY="
set /p "REPLY=%~1? [y/N] "
if /i "!REPLY!"=="y" exit /b 0
if /i "!REPLY!"=="yes" exit /b 0
echo   Skipped.
exit /b 1

:sizes
echo  Current sizes:
for %%D in (".venv" ".python" ".tools" ".uv-cache" ".models" "cache" "temp" "output") do call :one_size "%%~D"
exit /b 0

:one_size
if not exist "%~1" exit /b 0
set "BYTES=0"
for /f "tokens=3" %%S in ('dir /s /-c "%~1" 2^>nul ^| findstr /c:"File(s)"') do set "BYTES=%%S"
set /a MB=%BYTES%/1048576 2>nul
echo     %~1  !MB! MB
exit /b 0
