@echo off
rem Ply worker PC: double-click. Sets this PC up for its own GPU, pairs it with the Ply API (shows a code for the
rem Ply team to approve) and then runs the jobs the team assigns to it. Safe to run again.
rem Runs as you; Windows asks for admin rights only once, to turn on WSL (if it is not on yet).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipeline\local\start.ps1" %*
echo.
pause
