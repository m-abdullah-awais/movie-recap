# Project-scoped bootstrap for the Local Movie Recap Generator.
#
# Everything this script needs is either already on the machine or installed
# inside this project directory. Nothing is ever installed globally, to the user
# profile, to the registry, or to any shared location, and PATH is never
# modified. Several of the tools involved default to the user profile, so each
# of those defaults is redirected below before any work happens.
#
# The rule for every external program is the same: use the machine's copy when
# it has one, otherwise put a private copy in .tools and use that.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the built-in progress bar is slower than the download

# This script lives in scripts\, so the project root is one level up. uv finds
# pyproject.toml by searching upward from the working directory, so move there
# before doing anything, and the script then works from wherever it is called.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$tools = Join-Path $root '.tools'

# Redirect every cache and install location into the project.
$env:UV_PYTHON_INSTALL_DIR  = Join-Path $root '.python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $root '.venv'
$env:UV_CACHE_DIR           = Join-Path $root '.uv-cache'
$env:PIP_CACHE_DIR          = Join-Path $root '.uv-cache\pip'
$env:XDG_CACHE_HOME         = Join-Path $root '.uv-cache'
$env:HF_HOME                = Join-Path $root '.models'
$env:npm_config_cache       = Join-Path $root '.uv-cache\npm'
$env:npm_config_prefix      = Join-Path $tools 'claude'
$env:npm_config_global      = 'false'

# The default 30 second HTTP timeout is not enough on a slow connection, and
# eight parallel downloads make it worse by splitting the available bandwidth
# until every one of them times out. Fewer, more patient downloads succeed.
$env:UV_HTTP_TIMEOUT         = '300'
$env:UV_CONCURRENT_DOWNLOADS = '2'

$script:usedSystem = @()
$script:usedLocal = @()

function Write-Step($text) {
    Write-Host ''
    Write-Host "== $text" -ForegroundColor Cyan
}

function Write-Ok($text) { Write-Host "   $text" -ForegroundColor Green }
function Write-Info($text) { Write-Host "   $text" }

. (Join-Path $PSScriptRoot 'lib/fetch.ps1')

Write-Host ''
Write-Host '==============================================================' -ForegroundColor Cyan
Write-Host '   MOVIE RECAP GENERATOR, SETUP' -ForegroundColor Cyan
Write-Host '==============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host " Everything is installed inside:  $root"
Write-Host ' Nothing is installed globally, and your PATH is not changed.'
Write-Host ' Anything already on this machine is used as it is.'
Write-Host ''
Write-Host ' On a machine with none of this, the first run downloads about'
Write-Host ' 750 MB and can take a while on a slow connection. Anything that'
Write-Host ' finished is kept, so running it again resumes rather than restarts.'

New-Item -ItemType Directory -Force $tools | Out-Null

# --------------------------------------------------------------------------
# uv, which manages Python and the dependencies
# --------------------------------------------------------------------------
Write-Step 'uv'
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if ($uv) {
    $uvExe = $uv.Source
    Write-Ok "using the uv already on this machine: $uvExe"
    $script:usedSystem += 'uv'
} else {
    $uvExe = Join-Path $tools 'uv\uv.exe'
    if (-not (Test-Path $uvExe)) {
        $arch = if ((Get-Arch) -eq 'arm64') { 'aarch64' } else { 'x86_64' }
        Get-Archive "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip" `
                    (Join-Path $tools 'uv') 'uv, about 17 MB'
    }
    if (-not (Test-Path $uvExe)) { throw 'uv could not be installed into .tools' }
    Write-Ok "installed into the project: $uvExe"
    $script:usedLocal += 'uv'
}

# --------------------------------------------------------------------------
# Python 3.11 and the dependencies
# --------------------------------------------------------------------------
Write-Step 'Python 3.11 and dependencies'
Write-Info 'Python 3.11 specifically, because ctranslate2 publishes no wheels for newer versions.'
Write-Info 'A 3.11 already on this machine is used. Otherwise uv puts one in .python.'

# On a machine with nothing yet, --python-preference system uses a 3.11 the
# machine already has and only downloads a private copy into
# UV_PYTHON_INSTALL_DIR when there is none. Once an environment exists the
# preference is left alone, so re-running setup never rebuilds a working .venv
# around a different interpreter. Either way the environment itself is inside
# the project.
if ((Test-Path (Join-Path $root '.venv')) -or (Test-Path (Join-Path $root '.python'))) {
    & $uvExe sync --python 3.11
} else {
    & $uvExe sync --python 3.11 --python-preference system
}
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "uv sync did not produce $python" }
$base = (& $python -c "import sys; print(sys.base_prefix)")
if ($base -like "$root*") {
    Write-Ok 'dependencies installed, using a private Python in .python'
    $script:usedLocal += 'Python 3.11'
} else {
    Write-Ok "dependencies installed, using the Python already on this machine: $base"
    $script:usedSystem += 'Python 3.11'
}

# --------------------------------------------------------------------------
# ffmpeg and ffprobe, which do all of the video work
# --------------------------------------------------------------------------
Write-Step 'ffmpeg and ffprobe'
$haveFfmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)
$localFfmpeg = Join-Path $tools 'ffmpeg\bin\ffmpeg.exe'
if ($haveFfmpeg) {
    Write-Ok ("using the ffmpeg already on this machine: " + (Get-Command ffmpeg).Source)
    $script:usedSystem += 'ffmpeg'
} elseif (Test-Path $localFfmpeg) {
    Write-Ok "already in the project: $localFfmpeg"
    $script:usedLocal += 'ffmpeg'
} else {
    $name = if ((Get-Arch) -eq 'arm64') { 'ffmpeg-master-latest-winarm64-gpl' } else { 'ffmpeg-master-latest-win64-gpl' }
    Get-Archive "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/$name.zip" `
                (Join-Path $tools 'ffmpeg') 'ffmpeg, about 190 MB'
    if (-not (Test-Path $localFfmpeg)) { throw 'ffmpeg could not be installed into .tools' }
    Write-Ok "installed into the project: $localFfmpeg"
    $script:usedLocal += 'ffmpeg'
}

# --------------------------------------------------------------------------
# Claude Code, which every AI stage calls
# --------------------------------------------------------------------------
Write-Step 'Claude Code'
Write-Info 'Stages 4 and 5 run "claude -p" on your own subscription. No API key is used.'
$claude = Get-Command claude -ErrorAction SilentlyContinue
# The package ships its own executable. Everything prefers that over the .cmd
# shim npm writes beside it, because a .cmd hands every argument back to
# cmd.exe to re-parse, and the system prompt is passed as an argument.
$localClaude = Join-Path $tools 'claude\node_modules\@anthropic-ai\claude-code\bin\claude.exe'
if ($claude) {
    Write-Ok ("using the Claude Code already on this machine: " + $claude.Source)
    $script:usedSystem += 'Claude Code'
} elseif (Test-Path $localClaude) {
    Write-Ok "already in the project: $localClaude"
    $script:usedLocal += 'Claude Code'
} else {
    # npm is needed to install it. Use the machine's Node when it has one and it
    # is new enough, otherwise unpack a private copy into the project.
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    $nodeOk = $false
    if ($npm) {
        $nodeVersion = (& node --version) -replace '^v', ''
        $nodeOk = ([int]($nodeVersion -split '\.')[0]) -ge 22
        if (-not $nodeOk) {
            Write-Info "this machine has Node $nodeVersion, and Claude Code needs 22 or newer"
        }
    }
    if ($nodeOk) {
        $npmCmd = $npm.Source
        Write-Info ("using the Node already on this machine: " + (Get-Command node).Source)
        $script:usedSystem += 'Node'
    } else {
        $nodeDir = Join-Path $tools 'node'
        if (-not (Test-Path (Join-Path $nodeDir 'npm.cmd'))) {
            Get-Archive (Get-NodeUrl) $nodeDir 'Node, about 30 MB'
        }
        $npmCmd = Join-Path $nodeDir 'npm.cmd'
        if (-not (Test-Path $npmCmd)) { throw 'Node could not be installed into .tools' }
        $env:PATH = "$nodeDir;$env:PATH"
        Write-Ok "installed Node into the project: $nodeDir"
        $script:usedLocal += 'Node'
    }

    Write-Info 'installing Claude Code into the project, 70 MB to fetch, 240 MB on disk'
    New-Item -ItemType Directory -Force (Join-Path $tools 'claude') | Out-Null
    # --prefix keeps it in .tools\claude. It is never a global install, and npm
    # writes nothing to the user profile because its cache is redirected above.
    & $npmCmd install --prefix (Join-Path $tools 'claude') --no-fund --no-audit '@anthropic-ai/claude-code'
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $localClaude)) {
        Write-Host '   Claude Code could not be installed. Stages 4 and 5 will not run' -ForegroundColor Yellow
        Write-Host '   until it is. Everything else is ready.' -ForegroundColor Yellow
    } else {
        # The published executable is a stub until the package's own install
        # step fetches the real one, so it is run rather than merely found. An
        # npm configured with ignore-scripts would otherwise leave a file that
        # exists and does nothing.
        $version = (& $localClaude --version 2>&1)
        if ($LASTEXITCODE -ne 0) {
            Write-Host '   Claude Code installed but will not start. If npm here is set to' -ForegroundColor Yellow
            Write-Host '   ignore install scripts, that is why, and it has to be allowed.' -ForegroundColor Yellow
        } else {
            Write-Ok "installed into the project: $localClaude"
            Write-Info "reports itself as: $version"
            $script:usedLocal += 'Claude Code'
        }
    }
}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
Write-Step 'Models'
Write-Info 'The Kokoro narrator, about 340 MB, and the CLIP encoders, about 150 MB.'
& $python (Join-Path $PSScriptRoot 'analyze.py') fetch-models
if ($LASTEXITCODE -ne 0) {
    Write-Host '   Some models are missing. The pipeline still runs and degrades:' -ForegroundColor Yellow
    Write-Host '   without Kokoro the narrator is the Windows system voice, and without' -ForegroundColor Yellow
    Write-Host '   CLIP the footage is chosen by timing alone. Re-run this script later.' -ForegroundColor Yellow
}

# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
Write-Step 'Verifying containment'
& $python (Join-Path $PSScriptRoot 'analyze.py') doctor
$healthy = ($LASTEXITCODE -eq 0)

Write-Host ''
Write-Host '==============================================================' -ForegroundColor Cyan
if ($healthy) {
    Write-Host '   SETUP COMPLETE' -ForegroundColor Green
} else {
    Write-Host '   SETUP FINISHED WITH WARNINGS' -ForegroundColor Yellow
}
Write-Host '==============================================================' -ForegroundColor Cyan
Write-Host ''
if ($script:usedSystem.Count) {
    Write-Host (' Already on this machine, used as is:  ' + ($script:usedSystem -join ', '))
}
if ($script:usedLocal.Count) {
    Write-Host (' Installed inside this folder only:    ' + ($script:usedLocal -join ', '))
}
Write-Host ''
Write-Host ' Nothing was installed globally and your PATH was not changed.'
Write-Host ' Deleting this folder removes every trace of the tool.'
Write-Host ''
if (-not (Get-Command claude -ErrorAction SilentlyContinue) -and (Test-Path $localClaude)) {
    Write-Host ' One thing left to do. Claude Code needs you to sign in once:' -ForegroundColor Yellow
    Write-Host "   $localClaude"
    Write-Host ' Run it, sign in with your Claude subscription, then close it.'
    Write-Host ' The sign in is stored by Claude Code under your user profile,'
    Write-Host ' which is its own business and nothing to do with this project.'
    Write-Host ''
}
Write-Host ' Next: put a film in the input folder and double click Run.bat.'
Write-Host ''
if (-not $healthy) { exit 1 }
