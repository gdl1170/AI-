@echo off
title AI+ Installer
chcp 65001 >nul

echo ======================================================================
echo            AI+  --  Installazione automatica (Windows)
echo ======================================================================
echo.

REM --- Verifica / installa Python -----------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python non trovato. Scarico Python 3.11...
    echo.
    if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
        set PYURL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    ) else (
        set PYURL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-win32.exe
    )
    set PYINSTALLER=%TEMP%\python-installer.exe
    curl -fsSL %PYURL% -o %PYINSTALLER%
    if %errorlevel% neq 0 (
        echo [ERR] Download Python fallito. Scarica da: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    start /wait "" %PYINSTALLER% /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set USERPATH=%%b
    if defined USERPATH set PATH=%USERPATH%;%PATH%
)

python --version
if %errorlevel% neq 0 (
    echo [ERR] Python non trovato. Installalo da: https://www.python.org/downloads/
    echo      (Spunta: "Add Python to PATH")
    pause
    exit /b 1
)

echo [OK] Python trovato
echo.

REM --- Esegui setupAI+ (installa tutto e avvia server su http://localhost:8081)
python "%~dp0setupAI+"
pause
