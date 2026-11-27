# Ply on this Windows PC: a thin front for plylocal.sh inside WSL, so Windows paths work as they are.
# (Most people only need start.bat in the folder above. This is for running things by hand.)
#   pipeline\local\ply.ps1 doctor
#   pipeline\local\ply.ps1 run "C:\Users\me\Videos\lab 2" [-Name lab-2] [-Preset quick|standard|high] [-- plypipe flags...]
#   pipeline\local\ply.ps1 status [name] | results <name> [C:\dest] | jobs | worker | update | shell | forget
# (powershell -ExecutionPolicy Bypass -File pipeline\local\ply.ps1 ... if scripts are blocked on this PC)
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)][string]$Command = "help",
    [Parameter(Position = 1)][string]$Target = "",
    [Parameter(Position = 2)][string]$Dest = "",
    [string]$Name = "",
    [string]$Preset = "",
    [string]$Distro = "Ubuntu-24.04",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function To-Wsl([string]$p) {
    if (-not $p) { return "" }
    $full = (Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue)
    if (-not $full) { $full = [IO.Path]::GetFullPath($p) } else { $full = $full.Path }
    return (& wsl.exe -d $Distro -e wslpath -a ($full -replace '\\', '/')).Trim()
}
function Plylocal([string[]]$argv) {
    $home_ = (& wsl.exe -d $Distro -e sh -c 'echo $HOME').Trim()
    & wsl.exe -d $Distro -e bash "$home_/ply/pipeline/local/plylocal.sh" @argv
    exit $LASTEXITCODE
}

switch ($Command) {
    "setup"  { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "start.ps1") -NoWorker; exit $LASTEXITCODE }
    "doctor" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "doctor.ps1") -Distro $Distro
        Plylocal @("doctor")
    }
    "run" {
        if (-not $Target) { throw "usage: ply.ps1 run <clip or folder> [-Name N] [-Preset P] [-- plypipe flags]" }
        if (-not (Test-Path -LiteralPath $Target)) { throw "not found: $Target" }
        $argv = @("run", (To-Wsl $Target))
        if ($Name) { $argv += @("--name", $Name) }
        if ($Preset) { $argv += @("--preset", $Preset) }
        $argv += ($Rest | Where-Object { $_ -ne "--" })
        Plylocal $argv
    }
    "results" {
        if (-not $Target) { throw "usage: ply.ps1 results <name> [C:\dest]" }
        $argv = @("results", $Target)
        if ($Dest) { New-Item -ItemType Directory -Force -Path $Dest | Out-Null; $argv += (To-Wsl $Dest) }
        Plylocal $argv
    }
    "update" {
        # Windows clone from GitHub first, then the Ubuntu copy from the Windows clone
        $repo = (Resolve-Path (Join-Path $here "..\..")).Path
        & git -C $repo pull --ff-only
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Plylocal @("update")
    }
    "shell"  { & wsl.exe -d $Distro --cd "~"; exit $LASTEXITCODE }
    "forget" {
        Write-Host "Removes the Ubuntu distro $Distro and everything in it (jobs, install, caches), and %LOCALAPPDATA%\Ply."
        Write-Host "Results you copied out (ply.ps1 results) and this repo clone stay. A .wslconfig backup, if any, is restored."
        $a = Read-Host "Type FORGET to go on"
        if ($a -ne "FORGET") { Write-Host "nothing removed"; exit 1 }
        Unregister-ScheduledTask -TaskName "Ply worker" -Confirm:$false -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName "Ply setup (resume after reboot)" -Confirm:$false -ErrorAction SilentlyContinue
        & wsl.exe --unregister $Distro
        Remove-Item -Recurse -Force (Join-Path $env:LOCALAPPDATA "Ply") -ErrorAction SilentlyContinue
        $cfg = Join-Path $env:USERPROFILE ".wslconfig"
        if (Test-Path "$cfg.ply-backup") { Move-Item "$cfg.ply-backup" $cfg -Force } else { Remove-Item $cfg -ErrorAction SilentlyContinue }
        Write-Host "done. Tell the Ply team to revoke this PC's pairing. Also sign out of GitHub on this PC if you signed in (git credential manager / gh auth logout)."
    }
    { $_ -in @("status", "jobs", "worker") } {
        $argv = @($Command)
        if ($Target) { $argv += $Target }
        Plylocal $argv
    }
    default {
        Get-Content $MyInvocation.MyCommand.Path -TotalCount 6 | Select-Object -Skip 1 | ForEach-Object { $_ -replace '^# ?', '' }
    }
}
