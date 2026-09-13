# Project-scoped bootstrap for the Local Movie Recap Generator.
#
# Everything this script creates lives inside the project directory. Nothing is
# installed globally, to the user profile, or to any shared location. Several of
# the tools involved default to the user profile, so each of those defaults is
# redirected below before any work happens.

$ErrorActionPreference = 'Stop'

# This script lives in scripts\, so the project root is one level up. uv finds
# pyproject.toml by searching upward from the working directory, so move there
# before doing anything, and the script then works from wherever it is called.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Redirect every cache and install location into the project.
$env:UV_PYTHON_INSTALL_DIR  = Join-Path $root '.python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $root '.venv'
$env:UV_CACHE_DIR           = Join-Path $root '.uv-cache'
$env:PIP_CACHE_DIR          = Join-Path $root '.uv-cache\pip'
$env:XDG_CACHE_HOME         = Join-Path $root '.uv-cache'
$env:HF_HOME                = Join-Path $root '.models'

# The default 30 second HTTP timeout is not enough on a slow connection, and
# eight parallel downloads make it worse by splitting the available bandwidth
# until every one of them times out. Fewer, more patient downloads succeed.
$env:UV_HTTP_TIMEOUT        = '300'
$env:UV_CONCURRENT_DOWNLOADS = '2'

Write-Host 'Install locations (all inside the project):' -ForegroundColor Cyan
foreach ($v in 'UV_PYTHON_INSTALL_DIR','UV_PROJECT_ENVIRONMENT','UV_CACHE_DIR','PIP_CACHE_DIR','HF_HOME') {
    Write-Host ("  {0,-24} {1}" -f $v, [Environment]::GetEnvironmentVariable($v))
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not on PATH. Install uv first, then re-run this script.'
}

Write-Host ''
Write-Host 'Installing CPython 3.11 into .python ...' -ForegroundColor Cyan
Write-Host '(required because ctranslate2 has no wheels for the system Python 3.14)'
uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'uv python install 3.11 failed' }

Write-Host ''
Write-Host 'Creating .venv and installing dependencies ...' -ForegroundColor Cyan
Write-Host '(PyPI metadata is slow on this connection, allow a few minutes)'
uv sync --python 3.11 --managed-python
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }

Write-Host ''
Write-Host 'Verifying containment ...' -ForegroundColor Cyan
& (Join-Path $root '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'analyze.py') doctor
if ($LASTEXITCODE -ne 0) { throw 'doctor reported a problem' }

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Run the pipeline with:'
Write-Host '  .\Run.bat'
Write-Host 'or, for one stage at a time:'
Write-Host '  .\.venv\Scripts\python.exe scripts\analyze.py all "path\to\movie.mkv"'
