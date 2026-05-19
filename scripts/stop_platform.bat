@echo off
echo ==============================================
echo SOIL INTELLIGENCE SYSTEM - PLATFORM SHUTDOWN
echo ==============================================

echo Terminating API processes...
taskkill /FI "WINDOWTITLE eq Soil Intelligence API" /T /F >nul 2>&1

echo Terminating UI processes...
taskkill /FI "WINDOWTITLE eq Soil Intelligence UI" /T /F >nul 2>&1

echo Shutting down lingering python instances (if any)...
taskkill /IM python.exe /F >nul 2>&1

echo Platform Shutdown Complete.
pause
