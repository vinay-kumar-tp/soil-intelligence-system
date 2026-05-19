@echo off
echo ==============================================
echo SOIL INTELLIGENCE SYSTEM - PLATFORM STARTUP
echo ==============================================
cd %~dp0\..

echo [1/3] Running Environment Diagnostics...
.venv\Scripts\python.exe soil_ai_system\configs\startup_checks.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment Validation Failed! See reports/startup_validation_report.txt
    pause
    exit /b 1
)

echo [2/3] Starting Backend API...
start "Soil Intelligence API" cmd /c ".\run_backend.bat"

echo [3/3] Starting Frontend Dashboard...
start "Soil Intelligence UI" cmd /c ".\run_frontend.bat"

echo Platform started successfully.
pause
