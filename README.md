# ⚽🏀⚾🏈🏒 스포츠 점수 봇

각종 스포츠의 경기 결과, 리그 순위, 팀 승패 기록을 Discord에서 바로 확인하는 봇입니다.

## 지원 스포츠

| 스포츠 | 별칭 | 지원 리그 |
|--------|------|-----------|
| soccer (축구) | football, 축구, 풋볼 | Premier League, La Liga, Bundesliga, Serie A, Ligue 1, MLS, K League 1, UCL |
| baseball (야구) | 야구, mlb | MLB |
| basketball (농구) | 농구, nba | NBA, WNBA |
| football (미식축구) | 미식축구, nfl | NFL, NCAA |
| hockey (하키) | 아이스하키, 하키, nhl | NHL |

## 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `!scores <스포츠> [리그]` | 오늘의 경기 결과 및 스코어 | `!scores soccer premier` |
| `!standings <스포츠> [리그]` | 리그 순위표 (승-패-승률) | `!standings basketball nba` |
| `!team <스포츠> [리그] <팀이름>` | 팀 승패 기록 | `!team baseball mlb Yankees` |
| `!sports` | 지원 스포츠 및 리그 전체 목록 | `!sports` |
| `!help` | 도움말 | `!help` |

### 한국어 명령어

```
!경기결과 야구
!순위 축구 kleague
!팀기록 축구 laliga 레알마드리드
```

## 설치 및 실행

### 요구사항

- Python 3.10+
- Discord Bot Token

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 DISCORD_TOKEN 입력
```

### 실행

```bash
python bot.py
```

### Docker (선택)

```bash
docker build -t sports-bot .
docker run -e DISCORD_TOKEN=your_token sports-bot
```

## Discord Bot 설정

1. [Discord Developer Portal](https://discord.com/developers/applications) 에서 새 Application 생성
2. Bot 탭에서 Token 복사 → `.env` 파일의 `DISCORD_TOKEN`에 입력
3. **Privileged Gateway Intents** → `Message Content Intent` 활성화
4. OAuth2 → URL Generator에서 `bot` 스코프 + 아래 권한 선택:
   - Send Messages
   - Embed Links
   - Read Message History
5. 생성된 URL로 봇을 서버에 초대

## 데이터 출처

ESPN 공개 API를 사용합니다. 실시간 경기 데이터는 ESPN 서비스 정책에 따라 제공됩니다.
