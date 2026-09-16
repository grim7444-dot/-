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
set "WORKDIR=%~dp0"
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP_DIR%\trading-bot.lnk"
set "PS1=%TEMP%\install_trading_bot_shortcut.ps1"

if not exist "%TARGET%" (
    echo.
    echo   ERROR: start_bot_scheduled.bat not found next to this file.
    echo   Make sure install_startup.bat is still inside the trading-bot folder.
    echo.
    pause
    exit /b 1
)

REM Write a small PowerShell script to a temp file instead of using a
REM multi-line "powershell -Command ^" block -- the caret line-continuation
REM is fragile (a stray trailing space after ^ silently breaks it with no
REM error, which is why the window was closing instantly with nothing shown).
echo $s = (New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT%') > "%PS1%"
echo $s.TargetPath = '%TARGET%' >> "%PS1%"
echo $s.WorkingDirectory = '%WORKDIR%' >> "%PS1%"
echo $s.WindowStyle = 7 >> "%PS1%"
echo $s.Description = 'KRX trading bot - auto-start on login' >> "%PS1%"
echo $s.Save() >> "%PS1%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
set "PS_RESULT=%errorlevel%"
del "%PS1%" >nul 2>&1

if not "%PS_RESULT%"=="0" (
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
