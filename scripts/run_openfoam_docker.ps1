# Run Sinus_CFD nasal OpenFOAM case inside Docker (no WSL OpenFOAM install needed).
#
# Prerequisites:
#   1. Docker Desktop installed and RUNNING (whale icon in tray)
#   2. Case scaffolded: py -3.12 scripts\scaffold_openfoam_case.py --case VisibleHuman_Head
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_openfoam_docker.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_openfoam_docker.ps1 -ShellOnly
#   powershell -ExecutionPolicy Bypass -File scripts\run_openfoam_docker.ps1 -Image opencfd/openfoam-run:2312

param(
    [string]$Case = "VisibleHuman_Head",
    [string]$Image = "opencfd/openfoam-run:2412",
    [switch]$ShellOnly,
    [switch]$SkipPull
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CaseDir = Join-Path $Repo "foam\$Case"

Write-Host "=== Sinus_CFD OpenFOAM (Docker) ===" -ForegroundColor Cyan
Write-Host "Repo: $Repo"
Write-Host "Case: $CaseDir"
Write-Host "Image: $Image"

# Docker available?
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker not found on PATH." -ForegroundColor Red
    Write-Host "Install Docker Desktop, start it, then open a NEW PowerShell window."
    Write-Host "  winget install --id Docker.DockerDesktop -e --source winget"
    exit 1
}

docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker daemon not reachable. Start Docker Desktop and wait until it is ready." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $CaseDir "system\controlDict"))) {
    Write-Host "Scaffolding OpenFOAM case..." -ForegroundColor Yellow
    & py -3.12 (Join-Path $Repo "scripts\scaffold_openfoam_case.py") --case $Case
}

if (-not (Test-Path (Join-Path $CaseDir "constant\triSurface\solid_air_body.stl"))) {
    Write-Host "Missing solid_air_body.stl — exporting geometry + scaffolding..." -ForegroundColor Yellow
    & py -3.12 (Join-Path $Repo "scripts\export_openfoam_geometry.py") --case $Case
    & py -3.12 (Join-Path $Repo "scripts\scaffold_openfoam_case.py") --case $Case
}

if (-not $SkipPull) {
    Write-Host "Pulling OpenFOAM image (first time can be large)..." -ForegroundColor Cyan
    docker pull $Image
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pull failed for $Image - trying 2312..." -ForegroundColor Yellow
        $Image = "opencfd/openfoam-run:2312"
        docker pull $Image
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Trying community image openfoam/openfoam11-paraview510..." -ForegroundColor Yellow
            $Image = "openfoam/openfoam11-paraview510"
            docker pull $Image
        }
    }
}

# Convert Windows path for volume mount
$CaseMount = $CaseDir -replace "\\", "/"
# Docker Desktop on Windows accepts C:/Users/... form
if ($CaseMount -match "^([A-Za-z]):") {
    $drive = $Matches[1].ToLower()
    $CaseMount = "/$drive" + $CaseMount.Substring(2)
}

Write-Host "Mount: $CaseDir -> /case" -ForegroundColor Cyan

if ($ShellOnly) {
    Write-Host "Interactive OpenFOAM shell (cd /case already). Try: blockMesh" -ForegroundColor Green
    docker run --rm -it `
        -v "${CaseDir}:/case" `
        -w /case `
        $Image `
        bash
    exit $LASTEXITCODE
}

# Non-interactive Allrun inside container
$runScript = @'
set -e
cd /case
echo "PWD=$(pwd)"
ls -la system constant/triSurface 2>/dev/null | head -30

# Source OpenFOAM if not already (depends on image)
if [ -z "$WM_PROJECT_DIR" ]; then
  if [ -f /opt/openfoam*/etc/bashrc ]; then
    # shellcheck disable=SC1090
    . /opt/openfoam*/etc/bashrc 2>/dev/null || true
  fi
  if [ -f /usr/lib/openfoam/openfoam*/etc/bashrc ]; then
    . "$(ls /usr/lib/openfoam/openfoam*/etc/bashrc | head -1)"
  fi
  # ESI openfoam-run images usually set env via entrypoint
fi

echo "WM_PROJECT_DIR=${WM_PROJECT_DIR:-unset}"
command -v blockMesh
command -v snappyHexMesh
command -v simpleFoam

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

echo "DONE"
'@

# Write runner into case so line endings are less painful
$runnerWin = Join-Path $CaseDir "Allrun.docker"
# LF line endings
$runScript -replace "`r`n", "`n" | Set-Content -Path $runnerWin -NoNewline -Encoding utf8
# Ensure final newline
Add-Content -Path $runnerWin -Value "`n" -Encoding utf8

Write-Host "Running mesh + simpleFoam in Docker (can take a long time)..." -ForegroundColor Cyan
docker run --rm `
    -v "${CaseDir}:/case" `
    -w /case `
    $Image `
    bash -lc "sed -i 's/\r$//' /case/Allrun.docker 2>/dev/null; bash /case/Allrun.docker"

$code = $LASTEXITCODE
Write-Host "Docker run exit: $code"
if ($code -eq 0) {
    Write-Host "SUCCESS. Logs: foam\$Case\log.*" -ForegroundColor Green
    Write-Host "Results time directories under foam\$Case\ (e.g. 100, 200, ...)"
} else {
    Write-Host "Failed. Inspect foam\$Case\log.blockMesh / log.snappyHexMesh / log.simpleFoam" -ForegroundColor Yellow
}
exit $code
