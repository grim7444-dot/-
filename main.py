"""
스포츠 점수 봇 - Replit용 단일 파일 버전
환경변수 DISCORD_TOKEN 만 설정하면 바로 실행됩니다.
"""
import os, asyncio, aiohttp, discord
from discord.ext import commands
from datetime import date, datetime, timezone

# ── 설정 ──────────────────────────────────────────────────────────────────────

TOKEN  = os.environ["DISCORD_TOKEN"]   # Replit Secrets에서 설정
PREFIX = "!"

LEAGUES = {
    # 입력키워드 : (espn_sport, espn_league_id, 표시이름)
    "soccer":      ("soccer",   "eng.1",          "⚽ Premier League"),
    "축구":         ("soccer",   "eng.1",          "⚽ Premier League"),
    "premier":     ("soccer",   "eng.1",          "⚽ Premier League"),
    "laliga":      ("soccer",   "esp.1",          "⚽ La Liga"),
    "bundesliga":  ("soccer",   "ger.1",          "⚽ Bundesliga"),
    "seriea":      ("soccer",   "ita.1",          "⚽ Serie A"),
    "kleague":     ("soccer",   "kor.1",          "⚽ K League 1"),
    "ucl":         ("soccer",   "UEFA.CHAMPIONS", "⚽ UEFA Champions League"),
    "baseball":    ("baseball", "mlb",            "⚾ MLB"),
    "야구":         ("baseball", "mlb",            "⚾ MLB"),
    "mlb":         ("baseball", "mlb",            "⚾ MLB"),
    "kbo":         ("baseball", "kor.1",          "⚾ KBO 한국야구"),
    "npb":         ("baseball", "jpn.1",          "⚾ NPB 일본야구"),
    "basketball":  ("basketball","nba",           "🏀 NBA"),
    "농구":         ("basketball","nba",           "🏀 NBA"),
    "nba":         ("basketball","nba",           "🏀 NBA"),
    "football":    ("football", "nfl",            "🏈 NFL"),
    "미식축구":     ("football", "nfl",            "🏈 NFL"),
    "nfl":         ("football", "nfl",            "🏈 NFL"),
    "hockey":      ("hockey",   "nhl",            "🏒 NHL"),
    "하키":         ("hockey",   "nhl",            "🏒 NHL"),
    "nhl":         ("hockey",   "nhl",            "🏒 NHL"),
}

ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# ── Discord 봇 ────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)
_session: aiohttp.ClientSession | None = None

async def http():
    global _session
    if not _session or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

# ── API 호출 ──────────────────────────────────────────────────────────────────

async def fetch_scores(sport, league, target_date=None):
    s = await http()
    url = f"{ESPN}/{sport}/{league}/scoreboard"
    params = {"dates": target_date.strftime("%Y%m%d")} if target_date else {}
    async with s.get(url, params=params) as r:
        r.raise_for_status()
        data = await r.json()
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {})
        state  = status.get("type", {}).get("state", "pre")
        detail = status.get("type", {}).get("shortDetail", "")
        teams  = [
            {"name": c.get("team",{}).get("displayName","?"),
             "abbr": c.get("team",{}).get("abbreviation","?"),
             "score": c.get("score","-"),
             "winner": c.get("winner", False)}
            for c in comp.get("competitors", [])
        ]
        games.append({"date": ev.get("date",""), "state": state,
                      "detail": detail, "teams": teams})
    return games

async def fetch_standings(sport, league):
    s = await http()
    async with s.get(f"{ESPN}/{sport}/{league}/standings") as r:
        r.raise_for_status()
        data = await r.json()
    rows = []
    for child in data.get("children", []):
        entries = child.get("standings", {}).get("entries", []) or child.get("entries", [])
        for e in entries:
            team  = e.get("team", {})
            stats = {x["name"]: x.get("displayValue", x.get("value","")) for x in e.get("stats",[])}
            rows.append({
                "team":   team.get("displayName","?"),
                "abbr":   team.get("abbreviation","?"),
                "wins":   stats.get("wins",   stats.get("gamesWon","-")),
                "losses": stats.get("losses", stats.get("gamesLost","-")),
                "pct":    stats.get("winPercent", stats.get("pointsPercentage","-")),
            })
    return rows

# ── Embed 빌더 ────────────────────────────────────────────────────────────────

def _dt(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z","+00:00")).strftime("%m/%d %H:%M UTC")
    except Exception:
        return iso

def scores_embed(label, games):
    e = discord.Embed(title=f"{label} 경기 결과", color=0x3D6CFF,
                      timestamp=datetime.now(timezone.utc))
    if not games:
        e.description = "오늘 예정된 경기가 없습니다."
        return e
    for g in games:
        t = g["teams"]
        if len(t) < 2:
            continue
        away, home = t[0], t[1]
        if g["state"] == "post":
            winner = away if away["winner"] else (home if home["winner"] else None)
            line = f"**{away['abbr']} {away['score']}** vs **{home['score']} {home['abbr']}**"
            tag  = f"  ✅ {winner['name']} 승리" if winner else "  ⚖️ 무승부"
            e.add_field(name="✔️ 종료", value=f"{line}\n{tag}\n🕐 {_dt(g['date'])}", inline=False)
        elif g["state"] == "in":
            line = f"**{away['abbr']} {away['score']}** vs **{home['score']} {home['abbr']}**"
            e.add_field(name="▶️ 진행중", value=f"{line}\n🔴 {g['detail']}", inline=False)
        else:
            e.add_field(name="📅 예정", value=f"{away['name']} vs {home['name']}\n⏰ {_dt(g['date'])}", inline=False)
    e.set_footer(text="출처: ESPN")
    return e

def standings_embed(label, rows):
    e = discord.Embed(title=f"{label} 순위표", color=0x3D6CFF,
                      timestamp=datetime.now(timezone.utc))
    if not rows:
        e.description = "순위 데이터를 가져올 수 없습니다."
        return e
    lines = []
    for i, r in enumerate(rows[:20], 1):
        try: pct = f"{float(r['pct']):.3f}"
        except: pct = str(r['pct'])
        lines.append(f"`{i:>2}.` **{r['abbr']:<4}** {r['wins']}승 {r['losses']}패  ({pct})")
    e.description = "\n".join(lines)
    e.set_footer(text="출처: ESPN")
    return e

def help_embed():
    e = discord.Embed(title="⚽🏀⚾ 스포츠 점수 봇 도움말", color=0x3D6CFF)
    e.add_field(name="명령어", value=(
        "`!scores <스포츠/리그>` — 오늘 경기 결과\n"
        "`!standings <스포츠/리그>` — 리그 순위\n"
        "`!leagues` — 전체 키워드 목록"
    ), inline=False)
    e.add_field(name="예시", value=(
        "`!scores kbo`\n`!scores npb`\n`!scores mlb`\n"
        "`!scores 축구`\n`!standings nba`\n`!standings kleague`"
    ), inline=False)
    return e

# ── 명령어 ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"✅ 로그인: {bot.user}  |  서버 수: {len(bot.guilds)}")
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching, name="⚽⚾🏀 스포츠 경기"))

@bot.command(name="help", aliases=["도움말"])
async def cmd_help(ctx):
    await ctx.send(embed=help_embed())

@bot.command(name="leagues", aliases=["리그목록"])
async def cmd_leagues(ctx):
    keys = sorted(set(LEAGUES.keys()))
    e = discord.Embed(title="사용 가능한 키워드 목록", color=0x3D6CFF,
                      description="\n".join(f"`{k}` → {LEAGUES[k][2]}" for k in keys))
    await ctx.send(embed=e)

@bot.command(name="scores", aliases=["경기결과", "score", "스코어"])
async def cmd_scores(ctx, keyword: str = "soccer"):
    info = LEAGUES.get(keyword.lower())
    if not info:
        await ctx.send(f"❌ `{keyword}` 는 모르는 키워드입니다. `!leagues` 로 목록 확인하세요.")
        return
    sport, league, label = info
    async with ctx.typing():
        try:
            games = await fetch_scores(sport, league, date.today())
            await ctx.send(embed=scores_embed(label, games))
        except aiohttp.ClientResponseError as ex:
            await ctx.send(f"❌ API 오류 ({ex.status})")
        except Exception as ex:
            await ctx.send(f"❌ 오류: {ex}")

@bot.command(name="standings", aliases=["순위", "standing"])
async def cmd_standings(ctx, keyword: str = "soccer"):
    info = LEAGUES.get(keyword.lower())
    if not info:
        await ctx.send(f"❌ `{keyword}` 는 모르는 키워드입니다. `!leagues` 로 목록 확인하세요.")
        return
    sport, league, label = info
    async with ctx.typing():
        try:
            rows = await fetch_standings(sport, league)
            await ctx.send(embed=standings_embed(label, rows))
        except aiohttp.ClientResponseError as ex:
            await ctx.send(f"❌ API 오류 ({ex.status})")
        except Exception as ex:
            await ctx.send(f"❌ 오류: {ex}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ 인자 누락: `!help` 로 사용법을 확인하세요.")
    else:
        raise error

# ── 실행 ──────────────────────────────────────────────────────────────────────

bot.run(TOKEN)
