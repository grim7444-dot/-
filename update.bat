@echo off
REM ---------------------------------------------------------------------------
REM Double-click this to fetch the latest bot and run the connection check.
REM
REM It exists because typing multi-line commands by hand kept going wrong:
REM the wrong shell (PowerShell vs cmd), pasted example output, half-copied
REM lines. Double-clicking has none of those failure modes.
REM
REM The result is also written to result.txt, so it can be opened in Notepad
REM and copied from there. Copying out of a console window loses the start of
REM long output and invites pasting it straight back in as commands.
REM
REM .env is not in the archive, so your keys survive every update.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo.
echo ===============================================================
echo   Updating the bot from GitHub
echo ===============================================================
curl -L -o "%TEMP%\krxbot.zip" "https://github.com/grim7444-dot/-/archive/refs/heads/claude/krx-8stock-trading-bot.zip"
if errorlevel 1 goto :failed

tar -xf "%TEMP%\krxbot.zip" -C "%~dp0" --strip-components=1
if errorlevel 1 goto :failed

del "%TEMP%\krxbot.zip" >nul 2>&1
echo   Update complete.

echo.
echo ===============================================================
echo   Running the connection check (READ ONLY, no orders)
echo ===============================================================
python main.py check --live > "%~dp0result.txt" 2>&1
type "%~dp0result.txt"
goto :done

:failed
echo.
echo   Update FAILED. Check your internet connection and try again.
goto :end

:done
echo.
echo ===============================================================
echo   Saved to:  result.txt
echo   Open that file, select all, copy, and send it over.
echo ===============================================================
start "" notepad "%~dp0result.txt"

:end
echo.
pause
