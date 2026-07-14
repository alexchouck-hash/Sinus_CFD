#!/bin/bash
# Run inside Ubuntu WSL after first-time user setup:
#   bash /mnt/c/Users/houck/Documents/Sinus_CFD/scripts/install_openfoam_wsl.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
REPO="/mnt/c/Users/houck/Documents/Sinus_CFD"
CASE="$REPO/foam/VisibleHuman_Head"

echo "=== apt update / OpenFOAM install ==="
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl wget gnupg software-properties-common

# openfoam.org packages
if ! command -v simpleFoam >/dev/null 2>&1; then
  wget -O - https://dl.openfoam.org/gpg.key 2>/dev/null | sudo apt-key add - || true
  sudo add-apt-repository -y http://dl.openfoam.org/ubuntu || true
  sudo apt-get update -y || true
  sudo apt-get install -y openfoam11 || sudo apt-get install -y openfoam-default || true
fi

# ESI fallback
if ! command -v simpleFoam >/dev/null 2>&1; then
  curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash || true
  sudo apt-get update -y || true
  sudo apt-get install -y openfoam2412-default || sudo apt-get install -y openfoam2312-default || true
fi

# Source OpenFOAM
if [ -f /opt/openfoam11/etc/bashrc ]; then
  # shellcheck disable=SC1091
  source /opt/openfoam11/etc/bashrc
elif ls /usr/lib/openfoam/openfoam*/etc/bashrc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(ls /usr/lib/openfoam/openfoam*/etc/bashrc | head -1)"
elif ls /opt/openfoam*/etc/bashrc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(ls /opt/openfoam*/etc/bashrc | head -1)"
fi

echo "OpenFOAM: ${WM_PROJECT_DIR:-not set}"
command -v blockMesh
command -v snappyHexMesh
command -v simpleFoam

# Ensure case STLs exist (regenerate from Windows Python if missing)
if [ ! -f "$CASE/constant/triSurface/solid_air_body.stl" ]; then
  echo "triSurface missing — run on Windows first:"
  echo "  py -3.12 scripts/scaffold_openfoam_case.py --case VisibleHuman_Head"
  exit 1
fi

cd "$CASE"
chmod +x Allrun Allclean 2>/dev/null || true

# Prefer robust Allrun
cat > Allrun << 'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -f /opt/openfoam11/etc/bashrc ]; then source /opt/openfoam11/etc/bashrc
elif ls /usr/lib/openfoam/openfoam*/etc/bashrc >/dev/null 2>&1; then source "$(ls /usr/lib/openfoam/openfoam*/etc/bashrc | head -1)"
elif ls /opt/openfoam*/etc/bashrc >/dev/null 2>&1; then source "$(ls /opt/openfoam*/etc/bashrc | head -1)"
fi
echo "Using $WM_PROJECT_DIR"
blockMesh 2>&1 | tee log.blockMesh
surfaceFeatureExtract 2>&1 | tee log.surfaceFeatureExtract || true
snappyHexMesh -overwrite 2>&1 | tee log.snappyHexMesh
checkMesh 2>&1 | tee log.checkMesh
simpleFoam 2>&1 | tee log.simpleFoam
echo DONE
EOF
chmod +x Allrun

./Allrun
echo "Logs in $CASE/log.*"
