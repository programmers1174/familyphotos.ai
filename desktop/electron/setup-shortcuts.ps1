# Creates Desktop + Start Menu shortcuts so the app can be launched like a
# regular Windows app (double-click or Windows Search).
# Run once from the desktop\electron directory:
#   powershell -ExecutionPolicy Bypass -File setup-shortcuts.ps1

$appName   = "Family Photos"
$appDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$electronExe = Join-Path $appDir "node_modules\electron\dist\electron.exe"

if (-not (Test-Path $electronExe)) {
    Write-Error "electron.exe not found at: $electronExe`nRun 'npm install' first."
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell

function New-Shortcut($destination) {
    $lnk = $WshShell.CreateShortcut($destination)
    $lnk.TargetPath       = $electronExe
    $lnk.Arguments        = "`"$appDir`""
    $lnk.WorkingDirectory = $appDir
    $lnk.IconLocation     = "$electronExe,0"
    $lnk.Description      = "Family Photos - semantic photo search"
    $lnk.Save()
}

# Desktop shortcut
$desktop = [Environment]::GetFolderPath("Desktop")
New-Shortcut (Join-Path $desktop "$appName.lnk")
Write-Host "Created Desktop shortcut."

# Start Menu shortcut (makes it searchable in Windows Search)
$startMenu = Join-Path ([Environment]::GetFolderPath("ApplicationData")) `
             "Microsoft\Windows\Start Menu\Programs"
New-Shortcut (Join-Path $startMenu "$appName.lnk")
Write-Host "Created Start Menu shortcut."

Write-Host ""
Write-Host "Done! Search for '$appName' in the Windows search bar to launch the app."
