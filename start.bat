@echo off
rem Ply worker PC: double-click. Asks for admin once, sets this PC up for its own GPU, pairs it with the Ply API
rem (shows a code for the Ply team to approve) and then runs jobs the team assigns to it. Safe to run again.
setlocal
cd /d "%~dp0"
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs -WorkingDirectory '%~dp0'"
  exit /b
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipeline\local\start.ps1" %*
echo.
pause
