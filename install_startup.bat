@echo off
REM ---------------------------------------------------------------------------
REM Double-click ONCE to register the bot to auto-start whenever you log into
REM Windows. It creates a shortcut to start_bot_scheduled.bat inside your
REM Windows "Startup" folder (shell:startup) -- no manual copy/paste needed.
REM
REM Safe to run again later (e.g. after moving the bot folder): it just
REM overwrites the same shortcut with the new path.
REM
REM To UNDO: delete "trading-bot.lnk" from the Startup folder (Win+R,
REM type shell:startup, enter).
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "TARGET=%~dp0start_bot_scheduled.bat"
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP_DIR%\trading-bot.lnk"

if not exist "%TARGET%" (
    echo.
    echo   ERROR: start_bot_scheduled.bat not found next to this file.
    echo   Make sure install_startup.bat is still inside the trading-bot folder.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -Command ^
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT%');" ^
    "$s.TargetPath = '%TARGET%';" ^
    "$s.WorkingDirectory = '%~dp0';" ^
    "$s.WindowStyle = 7;" ^
    "$s.Description = 'KRX trading bot - auto-start on login';" ^
    "$s.Save()"

if errorlevel 1 (
    echo.
    echo   Shortcut creation FAILED. Send this window's output over.
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================================
echo   Done. The bot will now start automatically next time you log
echo   into Windows (a minimized window appears in the taskbar - check
echo   logs\startup_YYYYMMDD.log for the full result).
echo.
echo   To turn this off again: Win+R, type shell:startup, enter,
echo   then delete "trading-bot.lnk".
echo ===============================================================
echo.
pause
