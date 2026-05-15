@echo off
rem Use `python` (not `pythonw`) so the console stays open and logs are visible.
rem Once everything is dialed in, you can switch to `pythonw` to hide the window.
cd /d "%~dp0"
python main.py
pause
