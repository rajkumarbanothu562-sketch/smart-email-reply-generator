@echo off
title Smart Email Reply Generator
echo ======================================================================
echo           PROJECT 2: Smart Email Reply Generator
echo           DistilGPT2 + PyTorch + TorchServe + Amazon SageMaker
echo ======================================================================
echo.

set PYTHON_CMD=C:\Users\rajku\anaconda3\python.exe

if not exist "%PYTHON_CMD%" (
    set PYTHON_CMD=python
)

echo Using Python: %PYTHON_CMD%
echo.
echo Launching Web Application on http://127.0.0.1:5000 ...
echo Press Ctrl+C to terminate.
echo.

"%PYTHON_CMD%" run.py --port 5000
pause
