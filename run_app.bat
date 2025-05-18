@echo off
setlocal
title RVC MAKER

env\\Scripts\\python.exe main\\app\\app.py --open --allow_all_disk
echo.
pause
