@echo off
title Student Performance Analysis and Prediction System
cls
echo ===============================================================================
echo   STUDENT PERFORMANCE ANALYSIS AND PREDICTION SYSTEM (DATA MINING)
echo ===============================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)
python --version

echo.
echo [2/3] Checking required dependencies...
python -c "import streamlit, pandas, sklearn, matplotlib, seaborn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Dependencies not found. Automatically installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Could not install required dependencies.
        pause
        exit /b 1
    )
) else (
    echo All dependencies are satisfied!
)

echo.
echo [3/3] Launching Streamlit Web Dashboard...
echo The application will open in your default browser at: http://localhost:8501
echo To stop the application server, press Ctrl + C in this terminal window.
echo.

python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)
