# Ply local doctor: can this Windows PC run the pipeline, and how? Changes nothing on the machine.
#   powershell -ExecutionPolicy Bypass -File pipeline\local\doctor.ps1
# Writes %LOCALAPPDATA%\Ply\machine.json and prints the route:
#   wsl-train      NVIDIA GPU (sm 7.0+), WSL 2 available or installable  -> start.bat
#   needs-admin    WSL is not installed and this account cannot install it -> ask IT / the owner once
#   no-train       no usable NVIDIA GPU                                    -> not worth setting up; use Kaggle/Modal
# Works in Windows PowerShell 5.1 (every Windows 11) and PowerShell 7.
[CmdletBinding()]
param([string]$Distro = "Ubuntu-24.04", [switch]$Quiet)
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"   # wsl.exe's own output in UTF-8, not UTF-16
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

function Out-Line($label, $value, $state) {
    if ($Quiet) { return }
    $color = @{ ok = "Green"; warn = "Yellow"; bad = "Red"; info = "Gray" }[$state]
    Write-Host ("  {0,-14}" -f $label) -NoNewline
    Write-Host $value -ForegroundColor $color
}

$m = [ordered]@{}
$warnings = New-Object System.Collections.ArrayList
$blockers = New-Object System.Collections.ArrayList

if (-not $Quiet) { Write-Host "`nPly local doctor" -ForegroundColor Cyan }

# --- Windows, CPU, RAM, disk ---------------------------------------------------------------------------------
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$m.os = "$($os.Caption) build $($os.BuildNumber)"
$m.cpu = $cpu.Name.Trim()
$m.threads = [int]$cs.NumberOfLogicalProcessors
$m.ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
Out-Line "Windows" $m.os ($(if ([int]$os.BuildNumber -ge 22000) { "ok" } else { "warn" }))
Out-Line "CPU" "$($m.cpu) ($($m.threads) threads)" "info"
$ramState = if ($m.ramGB -ge 16) { "ok" } elseif ($m.ramGB -ge 12) { "warn" } else { "bad" }
Out-Line "RAM" "$($m.ramGB) GB" $ramState
if ($m.ramGB -lt 12) { [void]$warnings.Add("under 12 GB RAM: the camera solve of a long capture may run out of memory") }

$drives = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    [ordered]@{ drive = $_.DeviceID; freeGB = [math]::Round($_.FreeSpace / 1GB, 1); sizeGB = [math]::Round($_.Size / 1GB, 1) } }
$m.drives = @($drives)
$sys = $drives | Where-Object { $_.drive -eq $env:SystemDrive } | Select-Object -First 1
foreach ($d in $drives) {
    $state = if ($d.freeGB -ge 60) { "ok" } elseif ($d.freeGB -ge 30) { "warn" } else { "bad" }
    Out-Line "Disk $($d.drive)" "$($d.freeGB) GB free of $($d.sizeGB) GB" $state
}
# WSL's Linux disk lives on the system drive by default: installs ~25 GB, each job 10-40 GB
if ($sys -and $sys.freeGB -lt 50) { [void]$warnings.Add("$($sys.drive) has $($sys.freeGB) GB free: WSL lives there (install ~25 GB + 10-40 GB per job). start.ps1 -WslDrive D: puts it elsewhere") }

# --- GPU -----------------------------------------------------------------------------------------------------
$m.gpu = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = Get-Item "$env:SystemRoot\System32\nvidia-smi.exe" }
if ($smi) {
    $q = & $smi.Source --query-gpu=name,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($q) {
        $p = $q.Split(",") | ForEach-Object { $_.Trim() }
        $m.gpu = [ordered]@{ name = $p[0]; totalGB = [math]::Round([double]$p[1] / 1024, 2); freeGB = [math]::Round([double]$p[2] / 1024, 2);
                             sm = [double]$p[3]; driver = $p[4] }
    }
}
if ($m.gpu) {
    $g = $m.gpu
    $gstate = if ($g.sm -lt 7.0) { "bad" } elseif ($g.totalGB -lt 4) { "warn" } else { "ok" }
    Out-Line "GPU" "$($g.name), $($g.totalGB) GB ($($g.freeGB) free), sm $($g.sm)" $gstate
    $drv = [double]($g.driver.Split(".")[0])
    # CUDA 12.4 (the training environment) needs Windows driver 551.61 or newer
    $dstate = if ($drv -ge 552) { "ok" } elseif ($drv -ge 551) { "warn" } else { "bad" }
    Out-Line "Driver" $g.driver $dstate
    if ($drv -lt 551) { [void]$blockers.Add("NVIDIA driver $($g.driver) is too old for CUDA 12.4: install the current Game Ready / Studio driver from nvidia.com") }
    if ($g.sm -lt 7.0) { [void]$blockers.Add("compute capability $($g.sm) (pre-Turing): torch's CUDA 12 builds have no kernels for it") }
    # same estimate as plypipe/device.py: 2400 B per splat after a 1.2 GB reserve
    $cap = [math]::Floor(([math]::Max($g.freeGB - 1.2, 0.3)) * 1GB / 2400)
    $m.splatCeiling = $cap
    Out-Line "Splat ceiling" ("~{0:N2}M (estimate; the first run measures it)" -f ($cap / 1e6)) "info"
    $samNeed = if ($g.sm -ge 8) { 6 } else { 10 }
    if ($g.freeGB -lt $samNeed) { [void]$warnings.Add("SAM 3 (people masks, labels) needs ~$samNeed GB here: skipped automatically. Captures with people walking through: run the coverage stages on Kaggle, train here") }
    if ($g.freeGB -lt ($g.totalGB - 0.6)) { [void]$warnings.Add("the GPU already has $([math]::Round($g.totalGB - $g.freeGB, 1)) GB in use (desktop, browser, games): close them before training") }
} else {
    $vids = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ", "
    Out-Line "GPU" "no NVIDIA GPU found ($vids)" "bad"
    [void]$blockers.Add("no NVIDIA GPU: gsplat training needs CUDA (an AMD/Intel GPU cannot train this pipeline)")
}

# --- admin, virtualisation, WSL ------------------------------------------------------------------------------
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$m.admin = $admin
Out-Line "Admin" $(if ($admin) { "yes (this window)" } else { "no (this window)" }) "info"
$m.hypervisor = [bool]$cs.HypervisorPresent
$m.virtFirmware = [bool]$cpu.VirtualizationFirmwareEnabled
$virtOk = $m.hypervisor -or $m.virtFirmware
Out-Line "Virtualisation" $(if ($m.hypervisor) { "hypervisor running" } elseif ($m.virtFirmware) { "enabled in firmware" } else { "OFF in BIOS/UEFI" }) $(if ($virtOk) { "ok" } else { "bad" })

$m.wsl = $false; $m.distros = @()
$wslExe = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($wslExe) {
    $list = & wsl.exe --list --quiet 2>$null
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        $m.wsl = $true
        $m.distros = @($list | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
    }
}
$hasDistro = $m.distros -contains $Distro
$wslText = if ($hasDistro) { "installed, $Distro present" } elseif ($m.wsl) { "installed; $Distro not yet (start.bat adds it)" } else { "not installed" }
Out-Line "WSL" $wslText $(if ($hasDistro) { "ok" } elseif ($m.wsl) { "warn" } else { "warn" })
if (-not $m.wsl -and -not $admin) { [void]$blockers.Add("WSL is not installed and this window is not admin: start.bat asks for admin rights (if this account has none, the PC owner must allow it once)") }
if (-not $virtOk) { [void]$blockers.Add("virtualisation is off in the BIOS/UEFI (Intel VT-x / AMD-V): WSL 2 cannot start until it is on") }

# --- network (college proxies block some of these) -----------------------------------------------------------
$m.net = [ordered]@{}
foreach ($h in @("github.com", "pypi.org", "download.pytorch.org", "developer.download.nvidia.com", "huggingface.co", "nodejs.org")) {
    $ok = $false
    try {
        $r = Invoke-WebRequest -Uri "https://$h/" -Method Head -UseBasicParsing -TimeoutSec 12 -ErrorAction Stop
        $ok = $true
    } catch {
        # a 403/404 answer still proves the host is reachable
        if ($_.Exception.Response) { $ok = $true }
    }
    $m.net[$h] = $ok
    if (-not $ok) { [void]$warnings.Add("cannot reach $h (proxy or firewall?): the install downloads from it") }
}
$bad = @($m.net.Keys | Where-Object { -not $m.net[$_] })
Out-Line "Network" $(if ($bad.Count -eq 0) { "all download hosts reachable" } else { "blocked: $($bad -join ', ')" }) $(if ($bad.Count -eq 0) { "ok" } else { "bad" })

# --- route ---------------------------------------------------------------------------------------------------
$route = if (-not $m.gpu -or $m.gpu.sm -lt 7.0) { "no-train" } elseif ($blockers.Count -gt 0 -and -not $m.wsl -and -not $admin) { "needs-admin" } elseif ($blockers.Count -gt 0) { "blocked" } else { "wsl-train" }
$m.route = $route
$m.warnings = @($warnings)
$m.blockers = @($blockers)
$m.checkedAt = (Get-Date).ToString("s")

$dir = Join-Path $env:LOCALAPPDATA "Ply"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$m | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $dir "machine.json")

if (-not $Quiet) {
    Write-Host ""
    foreach ($b in $blockers) { Write-Host "  BLOCKER  $b" -ForegroundColor Red }
    foreach ($w in $warnings) { Write-Host "  note     $w" -ForegroundColor Yellow }
    Write-Host ""
    $next = @{
        "wsl-train"   = "Next: double-click start.bat";
        "needs-admin" = "Next: double-click start.bat and allow the admin prompt (only WSL itself needs it)";
        "blocked"     = "Fix the blockers above, then run this again";
        "no-train"    = "This PC cannot train. Use Kaggle / Modal for training (or plypipe --until occupancy here, slowly, on the CPU)" }[$route]
    Write-Host "  Route: $route" -ForegroundColor Cyan
    Write-Host "  $next"
    Write-Host "  Saved: $(Join-Path $dir 'machine.json')`n" -ForegroundColor DarkGray
}
if ($route -eq "wsl-train") { exit 0 } else { exit 2 }
