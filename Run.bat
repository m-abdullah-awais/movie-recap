@echo off
setlocal EnableDelayedExpansion
title Movie Recap Generator

rem ==========================================================
rem  Pick a film, then everything runs to completion on its own.
rem
rem  Usage:
rem    Run.bat        list the films in input\ and pick one
rem    Run.bat 2      start straight on the second film listed
rem
rem  The only question asked is which film. After that the whole
rem  pipeline runs unattended and writes the recap to output\.
rem ==========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

rem Project-scoped locations, matching setup.ps1. Nothing is ever installed or
rem cached outside this folder.
set "UV_PYTHON_INSTALL_DIR=%ROOT%\.python"
set "UV_PROJECT_ENVIRONMENT=%ROOT%\.venv"
set "UV_CACHE_DIR=%ROOT%\.uv-cache"
set "PIP_CACHE_DIR=%ROOT%\.uv-cache\pip"
set "XDG_CACHE_HOME=%ROOT%\.uv-cache"
set "HF_HOME=%ROOT%\.models"
set "UV_HTTP_TIMEOUT=300"
set "UV_CONCURRENT_DOWNLOADS=2"

set "PY=%ROOT%\.venv\Scripts\python.exe"
set "CHOICE=%~1"

cls
echo ==============================================================
echo    MOVIE RECAP GENERATOR
echo ==============================================================
echo.

rem ---- prerequisites ---------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 goto no_ffmpeg
if not exist "%PY%" goto no_venv

rem ---- list the films --------------------------------------------------------
if not exist "%ROOT%\input" md "%ROOT%\input" >nul 2>&1

set "COUNT=0"
for %%F in ("input\*.mkv" "input\*.mp4" "input\*.m4v" "input\*.avi" "input\*.mov" "input\*.webm" "input\*.wmv" "input\*.flv" "input\*.mpg" "input\*.mpeg" "input\*.m2ts" "input\*.ts") do call :add_film "%%~fF" "%%~nxF" "%%~zF"

if "%COUNT%"=="0" goto no_films

echo  Films in the input folder:
echo.
for /l %%N in (1,1,%COUNT%) do call :show_film %%N
echo.

if not "%CHOICE%"=="" goto validate

set "CHOICE="
set /p "CHOICE=Which film? Enter a number, or 0 to quit: "
if "%CHOICE%"=="" goto bye
if "%CHOICE%"=="0" goto bye

:validate
set "MOVIE=!FILM_%CHOICE%!"
if "!MOVIE!"=="" goto bad_choice

rem ---- run everything --------------------------------------------------------
cls
echo ==============================================================
echo    MOVIE RECAP GENERATOR
echo ==============================================================
echo.
echo  Film:  !NAME_%CHOICE%!
echo.
echo  Nine stages run now with no further questions. Each one prints
echo  when it starts and how long it took, so you can see what is
echo  done and what is left.
echo.
echo    1 ingest     read the film's dialogue
echo    2 proxy      read the film once, the longest stage
echo    3 scenemap   find the shot boundaries
echo    4 story      Claude works out the plot
echo    5 script     Claude writes the narration
echo    6 index      match shots to what the narration describes
echo    7 narrate    speak every line
echo    8 select     choose footage for each line
echo    9 render     encode the finished video
echo.
echo  Expect roughly 25 minutes for a 90 minute film on this machine,
echo  and about 30 for a 2 hour one. Anything already done is reused,
echo  so a repeat run is much faster.
echo.
echo  The recap is written to the output folder when it finishes.
echo  Press Ctrl+C at any point to stop; finished stages are kept.
echo.
echo --------------------------------------------------------------
echo.

set "STARTED=%TIME%"
"%PY%" analyze.py all "!MOVIE!"
set "CODE=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------
if not "%CODE%"=="0" goto failed

echo.
echo   FINISHED. Your recap is in the output folder:
echo.
for %%F in ("output\*.mp4") do echo     %%~nxF
echo.
call :confirm_open
goto bye

:failed
echo.
echo   Stopped before finishing, exit code %CODE%.
echo.
echo   The error is printed above. Everything completed so far is
echo   cached, so running this again resumes rather than restarting.
echo.
goto bye

rem ==========================================================
rem  problems
rem ==========================================================
:no_ffmpeg
echo  ffmpeg is not on PATH, and the tool cannot run without it.
echo  Install it, or set the FFMPEG and FFPROBE variables to its path.
echo.
pause
goto eof

:no_venv
echo  The project is not set up yet.
echo.
echo  Running setup now. It installs Python 3.11 and the dependencies
echo  inside this folder and downloads about 80 MB, which can take a
echo  while on a slow connection.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup.ps1"
if errorlevel 1 goto setup_failed
echo.
echo  Setup finished. Start this script again.
echo.
pause
goto eof

:setup_failed
echo.
echo  Setup did not finish. Re-running resumes from what it already
echo  downloaded, so try again.
echo.
pause
goto eof

:no_films
echo  No films found in the input folder.
echo.
echo  Put a movie in:  %ROOT%\input
echo.
echo  Any format ffmpeg reads works, such as mkv, mp4, avi or mov.
echo  Then start this script again.
echo.
call :confirm_open_input
goto eof

:bad_choice
echo.
echo  There is no film number %CHOICE%.
echo.
pause
goto eof

:bye
echo.
pause
goto eof

rem ==========================================================
rem  helpers
rem ==========================================================
:add_film
rem Called once per matching file. A pattern that matches nothing yields
rem nothing, so COUNT staying at zero means an empty folder.
set /a COUNT+=1
set "FILM_%COUNT%=%~1"
set "NAME_%COUNT%=%~2"
set /a "SIZE_%COUNT%=%~3/1048576"
exit /b 0

:show_film
set "N=%~1"
call set "NM=%%NAME_%N%%%"
call set "SZ=%%SIZE_%N%%%"
echo     [%N%]  !NM!  ^(!SZ! MB^)
exit /b 0

:confirm_open
set "REPLY="
set /p "REPLY=Open the output folder? [y/N] "
if /i "!REPLY!"=="y" start "" explorer "%ROOT%\output"
exit /b 0

:confirm_open_input
set "REPLY="
set /p "REPLY=Open the input folder now? [Y/n] "
if /i "!REPLY!"=="n" exit /b 0
start "" explorer "%ROOT%\input"
exit /b 0

:eof
endlocal
exit /b 0
