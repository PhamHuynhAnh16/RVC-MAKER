@echo off
setlocal
title RVC MAKER Tensorboard

env\\Scripts\\python.exe main/app/tensorboard.py --open
echo.
pause
