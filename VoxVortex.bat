@echo off
rem Double-click to open VoxVortex Desktop: starts the voice LLM and voice service, then the control panel in your browser.
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py %*
pause
