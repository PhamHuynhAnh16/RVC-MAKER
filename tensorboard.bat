@echo off
setlocal
title Vietnamese RVC Tensorboard

env\\Scripts\\python.exe main/app/run_tensorboard.py --open
echo.
pause