@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    set "PYTHON=python"
)

echo Creating local virtual environment (if needed)...
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Installing/updating required packages (including boto3 for AWS EC2)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check your internet connection and Python installation.
    pause
    exit /b 1
)

echo Starting GitHub Agent...
python -m streamlit run app.py --server.port 8501
pause
