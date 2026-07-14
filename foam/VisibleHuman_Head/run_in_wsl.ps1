# Run Allrun inside WSL (OpenFOAM must be installed in the distro)
$caseUnix = (wsl wslpath -a "C:\Users\houck\Documents\Sinus_CFD\foam\VisibleHuman_Head").Trim()
Write-Host "Case path in WSL: $caseUnix"
wsl -e bash -lc "cd '$caseUnix' && chmod +x Allrun Allclean && ./Allrun"
