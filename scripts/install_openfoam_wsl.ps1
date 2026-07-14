# Install Ubuntu (if needed) + OpenFOAM in WSL, then run the Sinus_CFD case.
# Prerequisites: reboot once after first "wsl --install" so Virtual Machine Platform is active.
#
# Usage (PowerShell):
#   cd C:\Users\houck\Documents\Sinus_CFD
#   powershell -ExecutionPolicy Bypass -File scripts\install_openfoam_wsl.ps1

$ErrorActionPreference = "Stop"
$RepoWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CaseWin = Join-Path $RepoWin "foam\VisibleHuman_Head"

Write-Host "=== Sinus_CFD OpenFOAM / WSL installer ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoWin"

# --- Ensure WSL distro exists ---
$distros = & wsl.exe -l -q 2>$null
if (-not $distros -or ($distros -join " ") -notmatch "Ubuntu") {
    Write-Host "Ubuntu not found. Installing Ubuntu (may need reboot after)..." -ForegroundColor Yellow
    & wsl.exe --install -d Ubuntu --no-launch
    Write-Host ""
    Write-Host "If this is the first WSL install, REBOOT Windows, then re-run this script." -ForegroundColor Yellow
    exit 2
}

# First launch may require creating a UNIX user interactively.
Write-Host "Checking Ubuntu..." -ForegroundColor Cyan
$probe = & wsl.exe -d Ubuntu -e bash -lc "echo OK && uname -a" 2>&1
if ($LASTEXITCODE -ne 0 -or ($probe -join " ") -notmatch "OK") {
    Write-Host "Ubuntu is installed but not initialized." -ForegroundColor Yellow
    Write-Host "Open 'Ubuntu' from the Start menu once, create a username/password, then re-run this script."
    Write-Host "Probe output: $probe"
    exit 3
}
Write-Host $probe

# --- Install OpenFOAM (openfoam.org packages for Ubuntu) ---
Write-Host "Installing OpenFOAM (this can take 10–30 minutes)..." -ForegroundColor Cyan

$installScript = @'
set -e
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl wget gnupg lsb-release software-properties-common

# Prefer openfoam.org packages when available for this Ubuntu release
. /etc/os-release
echo "Ubuntu codename: $VERSION_CODENAME"

if ! command -v simpleFoam >/dev/null 2>&1; then
  # Try OpenFOAM 11 (openfoam.org)
  curl -s https://dl.openfoam.org/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/openfoam.gpg 2>/dev/null || true
  # Fallback apt method used by many Ubuntu setups
  if ! grep -q openfoam /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key 2>/dev/null | apt-key add -" || true
    sudo add-apt-repository -y http://dl.openfoam.org/ubuntu || true
  fi
  sudo apt-get update -y || true
  sudo apt-get install -y openfoam11 || sudo apt-get install -y openfoam-default || sudo apt-get install -y openfoam
fi

# ESI OpenFOAM alternative if still missing
if ! command -v simpleFoam >/dev/null 2>&1 && ! ls /opt/openfoam* 2>/dev/null | head -1; then
  echo "Trying ESI openfoam.com packages..."
  curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash || true
  sudo apt-get update -y || true
  sudo apt-get install -y openfoam2412-default || sudo apt-get install -y openfoam2312-default || true
fi

echo "=== OpenFOAM detection ==="
command -v simpleFoam || true
ls /usr/lib/openfoam 2>/dev/null || true
ls /opt/openfoam* 2>/dev/null || true
ls /opt/OpenFOAM 2>/dev/null || true
'@

# Write install script into WSL home via stdin
$installScript | & wsl.exe -d Ubuntu -e bash -lc "cat > /tmp/install_openfoam.sh && chmod +x /tmp/install_openfoam.sh && bash /tmp/install_openfoam.sh"
if ($LASTEXITCODE -ne 0) {
    Write-Host "OpenFOAM package install returned code $LASTEXITCODE" -ForegroundColor Yellow
}

# --- Ensure foam case exists (STLs in metres) ---
Write-Host "Refreshing OpenFOAM case scaffold..." -ForegroundColor Cyan
& py -3.12 (Join-Path $RepoWin "scripts\scaffold_openfoam_case.py") --case VisibleHuman_Head

$caseUnix = (& wsl.exe -d Ubuntu wslpath -a $CaseWin).Trim()
Write-Host "Case path in WSL: $caseUnix"

# --- Patch Allrun to source whatever OpenFOAM was installed ---
$allrunFix = @'
set -e
cd "$(dirname "$0")"

source_of() {
  if [ -f /opt/openfoam11/etc/bashrc ]; then . /opt/openfoam11/etc/bashrc; return 0; fi
  if [ -f /opt/openfoam*/etc/bashrc ]; then . /opt/openfoam*/etc/bashrc; return 0; fi
  if [ -f /usr/lib/openfoam/openfoam*/etc/bashrc ]; then
    # shellcheck disable=SC1091
    . /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || . "$(ls -d /usr/lib/openfoam/openfoam*/etc/bashrc | head -1)"
    return 0
  fi
  if [ -f /usr/lib/openfoam/openfoam2312/etc/bashrc ]; then . /usr/lib/openfoam/openfoam2312/etc/bashrc; return 0; fi
  if [ -f "$HOME/OpenFOAM/OpenFOAM-11/etc/bashrc" ]; then . "$HOME/OpenFOAM/OpenFOAM-11/etc/bashrc"; return 0; fi
  # ESI layout
  if ls /usr/lib/openfoam/openfoam*/etc/bashrc >/dev/null 2>&1; then
    . "$(ls /usr/lib/openfoam/openfoam*/etc/bashrc | head -1)"
    return 0
  fi
  if [ -n "$WM_PROJECT_DIR" ]; then return 0; fi
  return 1
}

if ! source_of; then
  echo "ERROR: could not source OpenFOAM. Install openfoam11 or set WM_PROJECT_DIR."
  exit 1
fi

echo "Using OpenFOAM: $WM_PROJECT_DIR"
which blockMesh snappyHexMesh simpleFoam

echo "== blockMesh =="
blockMesh 2>&1 | tee log.blockMesh

echo "== surfaceFeatureExtract =="
surfaceFeatureExtract 2>&1 | tee log.surfaceFeatureExtract || true

echo "== snappyHexMesh =="
snappyHexMesh -overwrite 2>&1 | tee log.snappyHexMesh

echo "== checkMesh =="
checkMesh 2>&1 | tee log.checkMesh

echo "== simpleFoam =="
simpleFoam 2>&1 | tee log.simpleFoam

echo "Done."
'@

$allrunFix | & wsl.exe -d Ubuntu -e bash -lc "cat > '$caseUnix/Allrun' && chmod +x '$caseUnix/Allrun' '$caseUnix/Allclean'"

Write-Host "Running Allrun (mesh + simpleFoam). This can take a long time..." -ForegroundColor Cyan
& wsl.exe -d Ubuntu -e bash -lc "cd '$caseUnix' && ./Allrun"
$code = $LASTEXITCODE
Write-Host "Allrun exit code: $code"
if ($code -eq 0) {
    Write-Host "SUCCESS. View results in WSL with: cd $caseUnix && paraFoam -builtin" -ForegroundColor Green
} else {
    Write-Host "Allrun failed. Check logs in foam\VisibleHuman_Head\log.*" -ForegroundColor Yellow
}
exit $code
