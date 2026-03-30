#Requires -Version 5.1
<#
.SYNOPSIS
  Find the process listening on the familyphotos backend port (default 8765) via netstat, then kill it and its children.

.DESCRIPTION
  Uses netstat -ano output so you can see the same PIDs as in a manual netstat check.
  Override port with -Port or environment variable FAMILYPHOTOS_PORT.
#>
param(
  [int] $Port = $(if ($env:FAMILYPHOTOS_PORT) { [int]$env:FAMILYPHOTOS_PORT } else { 8765 })
)

$ErrorActionPreference = "Stop"

$pids = [System.Collections.Generic.HashSet[int]]::new()

foreach ($line in (netstat -ano)) {
  if ($line -notmatch "LISTENING") { continue }
  # Local address ends with :<port> before foreign column; avoid matching e.g. port 18765 via ':8765\s+'.
  if ($line -notmatch ":$Port\s+") { continue }
  if ($line -match "LISTENING\s+(\d+)\s*$") {
    [void]$pids.Add([int]$Matches[1])
  }
}

if ($pids.Count -eq 0) {
  Write-Host "No LISTENING socket found on port $Port (netstat -ano)." -ForegroundColor Yellow
  exit 0
}

foreach ($procId in $pids) {
  Write-Host "Killing process tree for PID $procId (port $Port)..." -ForegroundColor Cyan
  & taskkill.exe /PID $procId /T /F | Out-Host
}

Write-Host "Done." -ForegroundColor Green
