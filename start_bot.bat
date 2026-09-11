@echo off
REM 매일 아침 실행용: 최신 코드 받고 봇을 라이브로 시작한다.
REM 더블클릭으로 실행하거나, Windows 작업 스케줄러에 등록해서 매일
REM 특정 시간에 자동 실행할 수 있다.

cd /d "%~dp0"

echo ============================================
echo  1/3 최신 코드 받는 중...
echo ============================================
git pull origin claude/krx-8stock-trading-bot
if errorlevel 1 (
    echo.
    echo git pull 실패 - 인터넷 연결이나 git 상태를 확인하세요.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  2/3 STOPPED 상태 해제 (필요한 경우에만)
echo ============================================
python main.py resume

echo.
echo ============================================
echo  3/3 라이브 트레이딩 시작
echo ============================================
python main.py live --live

pause
