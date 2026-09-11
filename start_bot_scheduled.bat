@echo off
REM Windows 작업 스케줄러 전용 (무인 실행) -- 화면에 아무도 없어도 멈추지
REM 않도록 pause를 빼고, 결과를 logs\startup_YYYYMMDD.log 에 남긴다.

cd /d "%~dp0"
if not exist logs mkdir logs

set LOGFILE=logs\startup_%date:~0,4%%date:~5,2%%date:~8,2%.log

echo ============================================ >> "%LOGFILE%"
echo %date% %time%  자동 시작 시도 >> "%LOGFILE%"
echo ============================================ >> "%LOGFILE%"

git pull origin claude/krx-8stock-trading-bot >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo git pull 실패 >> "%LOGFILE%"
    exit /b 1
)

python main.py resume >> "%LOGFILE%" 2>&1

python main.py live --live >> "%LOGFILE%" 2>&1
