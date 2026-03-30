@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0kill-backend-on-port.ps1" %*
exit /b %ERRORLEVEL%
