# Download helpers shared by setup. Kept separate from setup.ps1 so they can be
# dot-sourced and exercised on their own, because unpacking someone else's
# archive layout is the part most likely to behave differently on another
# machine.

function Get-Arch {
    <#
        .SYNOPSIS
        x64 or arm64, as these projects name their Windows builds.

        Windows on ARM reports ARM64 here, and x64 builds of media tools run
        poorly under emulation, so the right archive is chosen rather than
        assumed.
    #>
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        return 'arm64'
    }
    return 'x64'
}

function Get-Archive {
    <#
        .SYNOPSIS
        Fetch a zip and unpack it to $Destination, which ends up holding the
        archive's contents rather than a version-named folder.

        Downloaded to a temporary name and moved into place only once complete,
        so an interrupted download is never mistaken for a finished install.
    #>
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Destination,
        [string]$Label = 'archive',
        [scriptblock]$Log = { param($m) Write-Host "   $m" }
    )

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force $parent | Out-Null

    $temp = Join-Path $env:TEMP ('recap-' + [IO.Path]::GetRandomFileName() + '.zip')
    $unpack = Join-Path $parent ('.unpack-' + [IO.Path]::GetRandomFileName())
    try {
        & $Log "downloading $Label"
        Invoke-WebRequest -Uri $Url -OutFile $temp -UseBasicParsing -TimeoutSec 1800

        & $Log "unpacking $Label"
        Expand-Archive -LiteralPath $temp -DestinationPath $unpack -Force

        # Most of these archives nest everything under a single folder named
        # for the version. Lift it out, so the installed layout does not change
        # every time the upstream version does.
        $entries = @(Get-ChildItem -LiteralPath $unpack)
        $source = if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) {
            $entries[0].FullName
        } else {
            $unpack
        }

        if (Test-Path $Destination) { Remove-Item -Recurse -Force $Destination }
        Move-Item -LiteralPath $source -Destination $Destination
    }
    finally {
        Remove-Item -Force $temp -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $unpack -ErrorAction SilentlyContinue
    }
}

function Get-NodeUrl {
    <#
        .SYNOPSIS
        The download URL for the current Node LTS build for this architecture.

        The LTS line publishes its exact version in its checksum file, so the
        version is read at install time rather than hardcoded and going stale.
    #>
    param([string]$Line = 'latest-v22.x')

    $base = "https://nodejs.org/dist/$Line"
    $index = (Invoke-WebRequest -Uri "$base/SHASUMS256.txt" -UseBasicParsing -TimeoutSec 300).Content
    $file = ([regex]::Match($index, 'node-v[\d.]+-win-' + (Get-Arch) + '\.zip')).Value
    if (-not $file) { throw "could not work out which Node build to download from $base" }
    return "$base/$file"
}
