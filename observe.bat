@echo off
REM ---------------------------------------------------------------------------
REM Double-click to WATCH the bot for a whole session without letting it order.
REM
REM It runs the real loop -- same cadence, same signals, same sizing, same
REM pre-trade checks -- against the simulated broker. Nothing is ever sent to
REM Kiwoom. Close the window or press Ctrl+C to stop.
REM
REM Use this for a few sessions before letting the bot trade for real.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo.
echo ===============================================================
echo   OBSERVATION MODE - no orders will be sent
echo   Press Ctrl+C to stop.
echo ===============================================================
echo.
python main.py live --dry-run --watch

echo.
echo ===============================================================
echo   Stopped. This window stays open until you press a key.
echo ===============================================================
pause
