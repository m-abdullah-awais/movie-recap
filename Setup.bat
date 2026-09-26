@echo off
setlocal
title Movie Recap Generator - Setup

rem ==========================================================
rem  One click install. Everything lands inside this folder.
rem
rem  Usage:
rem    Setup.bat              ask which AI engine, then install
rem    Setup.bat antigravity  install without asking
rem    Setup.bat claude       install without asking
rem
rem  Anything this machine already has is used as it is. Anything
rem  missing is installed into this folder only. Nothing is ever
rem  installed globally, no PATH is changed, and deleting this
rem  folder removes every trace of the tool.
rem ==========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

cls
echo ==============================================================
echo    MOVIE RECAP GENERATOR
echo ==============================================================
echo.
echo  This installs everything the tool needs into:
echo.
echo     %ROOT%
echo.
echo  What it does:
echo.
echo    - asks which AI engine writes the recap, Antigravity or
echo      Claude Code, and installs only the one you pick
echo    - uses any of uv, Python 3.11, ffmpeg or Node that this
echo      computer already has
echo    - installs the ones it does not have into this folder
echo    - downloads the narrator voice and the shot matching models
echo    - checks that nothing at all escaped this folder
echo.
echo  On a computer with none of this, the first run downloads about
echo  750 MB, so on a slow connection it takes a while. Stopping it
echo  and starting again resumes from what it already has.
echo.
echo  Nothing is installed globally on this computer.
echo.
echo --------------------------------------------------------------

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\setup.ps1" -Engine "%~1"
set "CODE=%ERRORLEVEL%"

echo.
if not "%CODE%"=="0" goto failed
echo  Setup finished. Put a film in the input folder, then run Run.bat.
echo.
pause
goto eof

:failed
echo  Setup did not finish cleanly, exit code %CODE%.
echo.
echo  The reason is printed above. Running this again picks up from
echo  what it already downloaded rather than starting over.
echo.
pause

:eof
endlocal
exit /b %CODE%
