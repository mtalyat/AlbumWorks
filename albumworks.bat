@echo off
echo Installing required packages...
pip install -r %~dp0requirements.txt 2>&1 >nul
if not %ERRORLEVEL%==0 (
    echo Failed to install required packages.
    pause
    exit /b %ERRORLEVEL%
)
cls
python %~dp0albumworks.py %*