# Ply worker PC, everything automatic (start.bat runs this as administrator). Safe to run again: done steps are skipped.
#   1 check the PC        2 WSL 2 (reboot once if Windows needs it; this resumes by itself after the reboot)
#   3 Ubuntu + user 'ply' 4 RAM for Linux     5 copy this folder in     6 install for THIS GPU (20-40 min once)
#   7 pair with the Ply API (shows a code for the team)                  8 run the worker (and, if you say yes, at logon)
# Flags: -NoWorker (set up and pair only), -NoAutostart, -Distro <name>, -WslDrive D: (Linux disk off C:)
[CmdletBinding()]
param([string]$Distro = "Ubuntu-24.04", [string]$WslDrive = "", [switch]$NoWorker, [switch]$NoAutostart, [switch]$Resumed)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
$state = Join-Path $env:LOCALAPPDATA "Ply"
New-Item -ItemType Directory -Force -Path $state | Out-Null
Start-Transcript -Path (Join-Path $state "start.log") -Append | Out-Null
$task = "Ply setup (resume after reboot)"

function Step($n, $t) { Write-Host "`n[$n/8] $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "`n$t" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 2 }
function Linux([string]$cmd, [switch]$Root) {
    # one bash command line inside Ubuntu; no double quotes in $cmd (Windows PowerShell 5.1 would mangle them)
    $u = if ($Root) { "root" } else { "ply" }
    & wsl.exe -d $Distro -u $u -e bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { throw "inside Ubuntu ($u): '$cmd' failed (exit $LASTEXITCODE)" }
}

if ($Resumed) { Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue }

Step 1 "checking this PC"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "doctor.ps1") -Distro $Distro
$m = Get-Content (Join-Path $state "machine.json") -Raw | ConvertFrom-Json
if ($m.route -eq "no-train") { Fail "This PC has no NVIDIA GPU that can train: there is nothing to set up. Thanks anyway!" }
$real = @($m.blockers | Where-Object { $_ -notmatch "WSL is not installed" })
if ($real.Count -gt 0) { Fail "Fix the BLOCKER lines above (driver update, or virtualisation on in the BIOS), then run start.bat again." }

Step 2 "WSL 2"
if (-not $m.wsl) {
    Write-Host "Installing WSL (a Windows feature)."
    & wsl.exe --install --no-distribution
    # come back by itself after the reboot: a one-time task at logon, removed when it runs
    $act = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$here\start.ps1`" -Resumed -Distro $Distro$(if ($WslDrive) { " -WslDrive $WslDrive" })$(if ($NoWorker) { ' -NoWorker' })$(if ($NoAutostart) { ' -NoAutostart' })"
    $trg = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    Register-ScheduledTask -TaskName $task -Action $act -Trigger $trg -RunLevel Highest -Force | Out-Null
    Write-Host "Windows needs a restart to finish WSL. Setup continues by itself after you log back in." -ForegroundColor Yellow
    $a = Read-Host "Restart now? [Y/n]"
    Stop-Transcript | Out-Null
    if ($a -notmatch '^(n|no)$') { Restart-Computer -Force }
    exit 0
}
try { & wsl.exe --update | Out-Null } catch { }
try { & wsl.exe --set-default-version 2 | Out-Null } catch { }
Write-Host "ok"

Step 3 "Ubuntu (user 'ply')"
if (-not ($m.distros -contains $Distro)) {
    & wsl.exe --install -d $Distro --no-launch
    if ($LASTEXITCODE -ne 0) { & wsl.exe --install -d $Distro --web-download --no-launch }
    if ($LASTEXITCODE -ne 0) { Fail "Could not install $Distro (network? Microsoft Store blocked?). See $state\start.log" }
    if ($WslDrive) {
        & wsl.exe --shutdown
        & wsl.exe --manage $Distro --move (Join-Path $WslDrive "WSL\$Distro")
    }
}
# older Store-packaged Ubuntu registers only on its first launch, which would ask for a user: register it as root instead
& wsl.exe -d $Distro -u root -e true
if ($LASTEXITCODE -ne 0) {
    $exe = ($Distro -replace '[^A-Za-z0-9]', '').ToLower() + ".exe"   # Ubuntu-24.04 -> ubuntu2404.exe
    & $exe install --root
    & wsl.exe -d $Distro -u root -e true
    if ($LASTEXITCODE -ne 0) { Fail "$Distro is installed but will not start. Try: wsl --install -d $Distro (then run start.bat again)" }
}
try { & wsl.exe --set-version $Distro 2 | Out-Null } catch { }
# a fixed Linux user, no questions: root makes it and makes it the default
Linux -Root "id ply >/dev/null 2>&1 || useradd -m -s /bin/bash ply; printf '[user]\ndefault=ply\n[interop]\nappendWindowsPath=false\n' > /etc/wsl.conf"
& wsl.exe --terminate $Distro | Out-Null
Write-Host "ok"

Step 4 "memory for Linux"
$cfg = Join-Path $env:USERPROFILE ".wslconfig"
$mem = [math]::Max(6, [math]::Floor($m.ramGB) - 4)
$have = if (Test-Path $cfg) { Get-Content $cfg -Raw } else { "" }
if ($have -notmatch "memory=${mem}GB") {
    if ($have -and -not (Test-Path "$cfg.ply-backup")) { Copy-Item $cfg "$cfg.ply-backup" }
    Set-Content -Path $cfg -Encoding ASCII -Value "[wsl2]`r`nmemory=${mem}GB`r`nswap=16GB`r`nprocessors=$($m.threads)`r`n"
    & wsl.exe --shutdown
    Write-Host "Linux gets $mem of $($m.ramGB) GB RAM + 16 GB swap ($cfg$(if ($have) { ', old file kept as .wslconfig.ply-backup' }))"
} else { Write-Host "ok ($mem GB)" }

Step 5 "copying Ply into Ubuntu"
$src = "$(& wsl.exe -d $Distro -u ply -e wslpath -a ($repo -replace '\\', '/'))".Trim()
# a fresh copy each time (code only: jobs live in ~/ply-work, the install in ~/.ply); CRLF stripped from the scripts in
# case this folder came through a Windows git with autocrlf
Linux "rm -rf ~/ply.new && cp -r '$src' ~/ply.new && rm -rf ~/ply && mv ~/ply.new ~/ply && find ~/ply/pipeline/local ~/ply/pipeline/scripts -name '*.sh' -exec sed -i 's/\r$//' {} + && echo copied"

Step 6 "installing for this GPU"
& wsl.exe -d $Distro -u ply -e bash -lc "test -f ~/.ply/env.sh && bash ~/ply/pipeline/local/plylocal.sh doctor >/dev/null 2>&1"
if ($LASTEXITCODE -eq 0) { Write-Host "already installed and working" }
else {
    Write-Host "20-40 minutes the first time, mostly downloads. Leave this window open."
    Linux -Root "bash /home/ply/ply/pipeline/local/install_local.sh system"
    Linux "bash ~/ply/pipeline/local/install_local.sh"
}

Step 7 "pairing with the Ply API"
& wsl.exe -d $Distro -u ply -e bash -lc "bash ~/ply/pipeline/local/plylocal.sh pair"
if ($LASTEXITCODE -ne 0) { Fail "Pairing did not finish. Run start.bat again to get a new code." }

if ($NoWorker) { Write-Host "`nSet up and paired. Start the worker with start.bat (or: $state\worker.cmd)." -ForegroundColor Green; Stop-Transcript | Out-Null; exit 0 }

Step 8 "worker"
# the launcher lives outside this folder, so deleting the download does not break it
$launcher = Join-Path $state "worker.ps1"
@"
# Ply worker: runs jobs assigned to this PC. Close this window (or Ctrl+C) to stop it.
`$host.UI.RawUI.WindowTitle = 'Ply worker (close to stop)'
Add-Type -Namespace Ply -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Ply.Power]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; keep the PC awake while this window is open (the display may still sleep)
`$env:WSL_UTF8 = '1'
& wsl.exe -d $Distro -u ply -e bash -lc 'bash ~/ply/pipeline/local/plylocal.sh worker'
"@ | Set-Content -Path $launcher -Encoding UTF8
"@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$launcher`"`r`n" | Set-Content -Path (Join-Path $state "worker.cmd") -Encoding ASCII

if (-not $NoAutostart) {
    $a = Read-Host "Start the Ply worker by itself whenever you log in to this PC? (a small window shows it is running) [y/N]"
    if ($a -match '^(y|yes)$') {
        $act = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Minimized -ExecutionPolicy Bypass -File `"$launcher`""
        $trg = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
        Register-ScheduledTask -TaskName "Ply worker" -Action $act -Trigger $trg -Force | Out-Null
        Write-Host "Autostart on (remove: pipeline\local\ply.ps1 forget, or Task Scheduler > 'Ply worker')."
    }
}
Stop-Transcript | Out-Null
Write-Host "`nThis PC is a Ply worker now. It runs the jobs the Ply team assigns to it; close this window to stop." -ForegroundColor Green
& powershell -NoProfile -ExecutionPolicy Bypass -File $launcher
