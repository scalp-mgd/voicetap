@echo off
rem Kill any python.exe whose command line contains main.py — voicetap is the
rem only such script in this folder, and personal machines rarely have other
rem `python main.py` running. Falls back to taskkill on the `voicetap` window
rem title (set by run.bat) so launches outside this folder still get caught.

powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
taskkill /F /FI "WINDOWTITLE eq voicetap" >nul 2>&1

echo voicetap stopped.
timeout /t 1 >nul
