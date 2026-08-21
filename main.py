"""
스포츠 분석 봇
환경변수:
  DISCORD_TOKEN   (필수)
  ODDS_API_KEY    (선택 - 언오버 배당, the-odds-api.com 무료 가입)
  SPORTS_API_KEY  (선택 - KBO/NPB 분석, api-sports.io 무료 가입)
"""
import os, aiohttp, discord
from discord.ext import commands
from datetime import date, datetime, timezone

TOKEN      = os.environ["DISCORD_TOKEN"]
ODDS_KEY   = os.environ.get("ODDS_API_KEY", "")
SPORTS_KEY = os.environ.get("SPORTS_API_KEY", "")
PREFIX     = "!"

ESPN      = "https://site.api.espn.com/apis/site/v2/sports"
MLB_API   = "https://statsapi.mlb.com/api/v1"
ODDS_API  = "https://api.the-odds-api.com/v4"
SPORTS_API = "https://v1.baseball.api-sports.io"

# api-sports.io 리그 ID
ASIA_LEAGUES = {
    "kbo": {"id": 6,  "name": "⚾ KBO 한국야구", "season": date.today().year},
    "npb": {"id": 2,  "name": "⚾ NPB 일본야구", "season": date.today().year},
}

LEAGUES = {
    "soccer":     ("soccer",     "eng.1",          "⚽ Premier League"),
    "축구":        ("soccer",     "eng.1",          "⚽ Premier League"),
    "premier":    ("soccer",     "eng.1",          "⚽ Premier League"),
    "laliga":     ("soccer",     "esp.1",          "⚽ La Liga"),
    "bundesliga": ("soccer",     "ger.1",          "⚽ Bundesliga"),
    "seriea":     ("soccer",     "ita.1",          "⚽ Serie A"),
    "kleague":    ("soccer",     "kor.1",          "⚽ K League 1"),
    "ucl":        ("soccer",     "UEFA.CHAMPIONS", "⚽ UCL"),
    "baseball":   ("baseball",   "mlb",            "⚾ MLB"),
    "야구":        ("baseball",   "mlb",            "⚾ MLB"),
    "mlb":        ("baseball",   "mlb",            "⚾ MLB"),
    "basketball": ("basketball", "nba",            "🏀 NBA"),
    "농구":        ("basketball", "nba",            "🏀 NBA"),
    "nba":        ("basketball", "nba",            "🏀 NBA"),
    "football":   ("football",   "nfl",            "🏈 NFL"),
    "미식축구":    ("football",   "nfl",            "🏈 NFL"),
    "nfl":        ("football",   "nfl",            "🏈 NFL"),
    "hockey":     ("hockey",     "nhl",            "🏒 NHL"),
    "하키":        ("hockey",     "nhl",            "🏒 NHL"),
    "nhl":        ("hockey",     "nhl",            "🏒 NHL"),
}

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)
_session = None

async def http():
    global _session
    if not _session or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

# ── ESPN 기본 스코어/순위 ──────────────────────────────────────────────────────

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
            {"name": c.get("team", {}).get("displayName", "?"),
             "abbr": c.get("team", {}).get("abbreviation", "?"),
             "score": c.get("score", "-"),
             "winner": c.get("winner", False)}
            for c in comp.get("competitors", [])
        ]
        games.append({"date": ev.get("date", ""), "state": state,
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
            stats = {x["name"]: x.get("displayValue", x.get("value", "")) for x in e.get("stats", [])}
            rows.append({
                "team":   team.get("displayName", "?"),
                "abbr":   team.get("abbreviation", "?"),
                "wins":   stats.get("wins",   stats.get("gamesWon", "-")),
                "losses": stats.get("losses", stats.get("gamesLost", "-")),
                "pct":    stats.get("winPercent", stats.get("pointsPercentage", "-")),
            })
    return rows

# ── MLB Stats API ─────────────────────────────────────────────────────────────

async def get_mlb_schedule():
    s = await http()
    today = date.today().strftime("%Y-%m-%d")
    params = {
        "sportId": 1,
        "date": today,
        "hydrate": "probablePitcher,team,linescore",
    }
    async with s.get(f"{MLB_API}/schedule", params=params) as r:
        r.raise_for_status()
        data = await r.json()
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {})
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_p = home.get("probablePitcher", {})
            away_p = away.get("probablePitcher", {})
            games.append({
                "status": status.get("abstractGameState", "Preview"),
                "home": {
                    "id":    home.get("team", {}).get("id"),
                    "name":  home.get("team", {}).get("name", "?"),
                    "abbr":  home.get("team", {}).get("abbreviation", "?"),
                    "score": home.get("score"),
                },
                "away": {
                    "id":    away.get("team", {}).get("id"),
                    "name":  away.get("team", {}).get("name", "?"),
                    "abbr":  away.get("team", {}).get("abbreviation", "?"),
                    "score": away.get("score"),
                },
                "home_pitcher": {"id": home_p.get("id"), "name": home_p.get("fullName", "미정")} if home_p else None,
                "away_pitcher": {"id": away_p.get("id"), "name": away_p.get("fullName", "미정")} if away_p else None,
                "startTime": game.get("gameDate", ""),
            })
    return games

async def get_pitcher_stats(person_id):
    if not person_id:
        return None
    s = await http()
    params = {"stats": "season", "group": "pitching", "season": date.today().year}
    async with s.get(f"{MLB_API}/people/{person_id}/stats", params=params) as r:
        if r.status != 200:
            return None
        data = await r.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    stat = splits[0].get("stat", {})
    return {
        "era": stat.get("era", "-"),
        "whip": stat.get("whip", "-"),
        "wins": stat.get("wins", 0),
        "losses": stat.get("losses", 0),
        "strikeOuts": stat.get("strikeOuts", 0),
        "inningsPitched": stat.get("inningsPitched", "0"),
    }

async def get_team_batting_stats(team_id):
    if not team_id:
        return None
    s = await http()
    params = {"stats": "season", "group": "hitting", "season": date.today().year}
    async with s.get(f"{MLB_API}/teams/{team_id}/stats", params=params) as r:
        if r.status != 200:
            return None
        data = await r.json()
    for sg in data.get("stats", []):
        if sg.get("group", {}).get("displayName") == "hitting":
            splits = sg.get("splits", [])
            if splits:
                stat = splits[0].get("stat", {})
                return {
                    "avg": stat.get("avg", "-"),
                    "ops": stat.get("ops", "-"),
                    "homeRuns": stat.get("homeRuns", 0),
                }
    return None

# ── KBO / NPB (api-sports.io) ─────────────────────────────────────────────────

async def get_asian_games(league_id):
    if not SPORTS_KEY:
        return None
    s = await http()
    headers = {"x-apisports-key": SPORTS_KEY}
    params  = {
        "league": league_id,
        "season": date.today().year,
        "date":   date.today().strftime("%Y-%m-%d"),
    }
    async with s.get(f"{SPORTS_API}/games", headers=headers, params=params) as r:
        if r.status != 200:
            return None
        data = await r.json()
    games = []
    for g in data.get("response", []):
        teams  = g.get("teams", {})
        scores = g.get("scores", {})
        status = g.get("status", {})
        pitchers = g.get("pitchers", {})
        games.append({
            "home": teams.get("home", {}).get("name", "?"),
            "away": teams.get("away", {}).get("name", "?"),
            "home_score": scores.get("home", {}).get("total", "-"),
            "away_score": scores.get("away", {}).get("total", "-"),
            "status": status.get("long", "Scheduled"),
            "home_pitcher": pitchers.get("home", {}).get("name"),
            "away_pitcher": pitchers.get("away", {}).get("name"),
            "home_era":  pitchers.get("home", {}).get("era"),
            "away_era":  pitchers.get("away", {}).get("era"),
            "home_wins": pitchers.get("home", {}).get("wins"),
            "home_losses": pitchers.get("home", {}).get("losses"),
            "away_wins": pitchers.get("away", {}).get("wins"),
            "away_losses": pitchers.get("away", {}).get("losses"),
        })
    return games

async def get_asian_team_stats(league_id):
    if not SPORTS_KEY:
        return {}
    s = await http()
    headers = {"x-apisports-key": SPORTS_KEY}
    params  = {"league": league_id, "season": date.today().year}
    async with s.get(f"{SPORTS_API}/standings", headers=headers, params=params) as r:
        if r.status != 200:
            return {}
        data = await r.json()
    stats = {}
    for entry in data.get("response", []):
        team = entry.get("team", {}).get("name", "")
        w    = entry.get("won", 0)
        l    = entry.get("lost", 0)
        pct  = entry.get("win_pct", "-")
        stats[team] = {"wins": w, "losses": l, "pct": pct}
    return stats

# ── The Odds API ──────────────────────────────────────────────────────────────

async def get_odds_data():
    if not ODDS_KEY:
        return {}
    s = await http()
    params = {
        "apiKey": ODDS_KEY,
        "regions": "us",
        "markets": "totals,h2h",
        "oddsFormat": "american",
    }
    try:
        async with s.get(f"{ODDS_API}/sports/baseball_mlb/odds", params=params) as r:
            if r.status != 200:
                return {}
            data = await r.json()
        odds_map = {}
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            totals = None
            h2h    = {}
            for bm in event.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "totals" and totals is None:
                        t = {}
                        for o in market.get("outcomes", []):
                            if o["name"] == "Over":
                                t["line"] = o.get("point")
                                t["over"] = o.get("price")
                            elif o["name"] == "Under":
                                t["under"] = o.get("price")
                        totals = t
                    elif market.get("key") == "h2h":
                        for o in market.get("outcomes", []):
                            if o["name"] == home:
                                h2h["home"] = o.get("price")
                            elif o["name"] == away:
                                h2h["away"] = o.get("price")
                if totals:
                    break
            odds_map[home] = {"totals": totals, "h2h": h2h, "home": home, "away": away}
        return odds_map
    except Exception:
        return {}

# ── 분석 로직 ─────────────────────────────────────────────────────────────────

def _fmt_odds(price):
    if price is None:
        return "?"
    return f"+{price}" if price > 0 else str(price)

def analyze_matchup(home_pstats, away_pstats, home_bat, away_bat):
    lines = []
    try:
        he = float(home_pstats["era"]) if home_pstats and home_pstats["era"] != "-" else 4.50
        ae = float(away_pstats["era"]) if away_pstats and away_pstats["era"] != "-" else 4.50
        diff = ae - he
        if abs(diff) < 0.30:
            lines.append("🔵 투수 매치업: 엇비슷")
        elif diff > 0:
            lines.append(f"🏠 홈팀 투수 우위 (ERA 차 {abs(diff):.2f})")
        else:
            lines.append(f"✈️ 원정팀 투수 우위 (ERA 차 {abs(diff):.2f})")
    except (TypeError, ValueError):
        pass
    try:
        ho = float(home_bat["ops"]) if home_bat and home_bat["ops"] != "-" else 0.700
        ao = float(away_bat["ops"]) if away_bat and away_bat["ops"] != "-" else 0.700
        diff = ho - ao
        if abs(diff) < 0.020:
            lines.append("🔵 타선: 엇비슷")
        elif diff > 0:
            lines.append(f"🏠 홈팀 타선 우위 (OPS 차 {abs(diff):.3f})")
        else:
            lines.append(f"✈️ 원정팀 타선 우위 (OPS 차 {abs(diff):.3f})")
    except (TypeError, ValueError):
        pass
    try:
        he = float(home_pstats["era"]) if home_pstats and home_pstats["era"] != "-" else 4.50
        ae = float(away_pstats["era"]) if away_pstats and away_pstats["era"] != "-" else 4.50
        ho = float(home_bat["ops"])    if home_bat and home_bat["ops"] != "-"    else 0.700
        ao = float(away_bat["ops"])    if away_bat and away_bat["ops"] != "-"    else 0.700
        hs = (ae - 4.0) * 10 + (ho - 0.700) * 100
        as_ = (he - 4.0) * 10 + (ao - 0.700) * 100
        if abs(hs - as_) < 3:
            lines.append("⚖️ **종합 예측: 박빙**")
        elif hs > as_:
            lines.append("🏠 **종합 예측: 홈팀 유리**")
        else:
            lines.append("✈️ **종합 예측: 원정팀 유리**")
    except (TypeError, ValueError):
        pass
    return "\n".join(lines) if lines else "분석 데이터 부족"

# ── Embed 빌더 ────────────────────────────────────────────────────────────────

def _dt(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%m/%d %H:%M UTC")
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

async def build_mlb_embed(games, odds_map):
    e = discord.Embed(title="⚾ MLB 오늘 경기 분석",
                      color=0x1E90FF, timestamp=datetime.now(timezone.utc))
    if not games:
        e.description = "오늘 MLB 경기가 없습니다."
        return e
    for game in games[:6]:
        home, away = game["home"], game["away"]
        hp, ap = game.get("home_pitcher"), game.get("away_pitcher")
        hp_stats = ap_stats = hb = ab = None
        try:
            if hp and hp.get("id"): hp_stats = await get_pitcher_stats(hp["id"])
            if ap and ap.get("id"): ap_stats = await get_pitcher_stats(ap["id"])
            if home.get("id"):      hb = await get_team_batting_stats(home["id"])
            if away.get("id"):      ab = await get_team_batting_stats(away["id"])
        except Exception:
            pass

        st = game["status"]
        if st in ("Final", "Game Over"):
            title = f"✔️ 종료: {away['abbr']} {away['score']} - {home['score']} {home['abbr']}"
        elif st in ("In Progress", "Live"):
            title = f"▶️ 진행중: {away['abbr']} {away['score']} - {home['score']} {home['abbr']}"
        else:
            title = f"📅 {away['name']} @ {home['name']}  ⏰ {_dt(game['startTime'])}"

        # 투수
        def pline(p, ps, side):
            name = p["name"] if p else "미정"
            if ps:
                return (f"{side} **{name}** — {ps['wins']}승{ps['losses']}패 "
                        f"ERA {ps['era']} WHIP {ps['whip']} K {ps['strikeOuts']}")
            return f"{side} **{name}** — 시즌 성적 없음"

        p_text = pline(ap, ap_stats, "✈️") + "\n" + pline(hp, hp_stats, "🏠")

        # 타선
        bat_lines = []
        if ab: bat_lines.append(f"✈️ {away['abbr']}: 타율 {ab['avg']}  OPS {ab['ops']}  HR {ab['homeRuns']}")
        if hb: bat_lines.append(f"🏠 {home['abbr']}: 타율 {hb['avg']}  OPS {hb['ops']}  HR {hb['homeRuns']}")
        bat_text = "\n".join(bat_lines) if bat_lines else "타격 데이터 없음"

        analysis = analyze_matchup(hp_stats, ap_stats, hb, ab)

        # 언오버
        ou_text = ""
        for h_key, od in odds_map.items():
            if (home["name"].lower() in h_key.lower() or h_key.lower() in home["name"].lower()):
                t = od.get("totals")
                if t:
                    ou_text = (f"📊 언오버 라인: **{t.get('line')}**  "
                               f"오버 {_fmt_odds(t.get('over'))} / 언더 {_fmt_odds(t.get('under'))}")
                break

        value = f"**[선발 투수]**\n{p_text}\n\n**[팀 타격]**\n{bat_text}\n\n**[분석]**\n{analysis}"
        if ou_text:
            value += f"\n\n{ou_text}"
        e.add_field(name=title, value=value[:1024], inline=False)

    e.set_footer(text="출처: MLB Stats API" + (" | The Odds API" if odds_map else ""))
    return e

def build_asian_embed(league_name, games, team_stats):
    e = discord.Embed(title=f"{league_name} 오늘 경기 분석",
                      color=0xFF4500, timestamp=datetime.now(timezone.utc))
    if not games:
        e.description = "오늘 예정된 경기가 없습니다."
        return e
    for g in games:
        st = g["status"]
        if "Final" in st or "Finished" in st:
            title = f"✔️ 종료: {g['away']} {g['away_score']} - {g['home_score']} {g['home']}"
        elif "In" in st or "Progress" in st or "Live" in st:
            title = f"▶️ 진행중: {g['away']} {g['away_score']} - {g['home_score']} {g['home']}"
        else:
            title = f"📅 {g['away']} @ {g['home']}"

        lines = []
        # 투수
        hp, ap = g.get("home_pitcher"), g.get("away_pitcher")
        if ap:
            rec = f"{g['away_wins']}승{g['away_losses']}패" if g.get("away_wins") is not None else ""
            era = f"ERA {g['away_era']}" if g.get("away_era") else ""
            lines.append(f"✈️ **{ap}** {rec} {era}".strip())
        if hp:
            rec = f"{g['home_wins']}승{g['home_losses']}패" if g.get("home_wins") is not None else ""
            era = f"ERA {g['home_era']}" if g.get("home_era") else ""
            lines.append(f"🏠 **{hp}** {rec} {era}".strip())

        # 팀 시즌 기록
        for team, side in [(g["away"], "✈️"), (g["home"], "🏠")]:
            ts = team_stats.get(team)
            if ts:
                lines.append(f"{side} {team}: {ts['wins']}승 {ts['losses']}패 (승률 {ts['pct']})")

        # ERA 기반 간단 예측
        try:
            he = float(g["home_era"]) if g.get("home_era") else 4.50
            ae = float(g["away_era"]) if g.get("away_era") else 4.50
            diff = ae - he
            if abs(diff) < 0.30:
                lines.append("⚖️ **예측: 박빙**")
            elif diff > 0:
                lines.append("🏠 **예측: 홈팀 투수 우위**")
            else:
                lines.append("✈️ **예측: 원정팀 투수 우위**")
        except (TypeError, ValueError):
            pass

        e.add_field(name=title, value="\n".join(lines) or "데이터 없음", inline=False)

    e.set_footer(text="출처: API-Sports")
    return e

def build_odds_embed(odds_map):
    e = discord.Embed(title="⚾ MLB 오늘 언오버 배당", color=0x00CC44,
                      timestamp=datetime.now(timezone.utc))
    if not odds_map:
        e.description = "배당 데이터가 없습니다."
        return e
    for _, od in list(odds_map.items())[:12]:
        t = od.get("totals")
        h = od.get("h2h", {})
        if not t:
            continue
        home_line = _fmt_odds(h.get("home"))
        away_line = _fmt_odds(h.get("away"))
        ou = f"라인 **{t.get('line')}** | 오버 {_fmt_odds(t.get('over'))} / 언더 {_fmt_odds(t.get('under'))}"
        ml = f"머니라인: 홈 {home_line} / 원정 {away_line}" if h else ""
        e.add_field(
            name=f"{od['away']} @ {od['home']}",
            value=ou + (f"\n{ml}" if ml else ""),
            inline=False,
        )
    e.set_footer(text="출처: The Odds API | 참고용 정보, 도박 권장 아님")
    return e

def help_embed():
    e = discord.Embed(title="⚾🏀⚽ 스포츠 분석 봇 도움말", color=0x3D6CFF)
    e.add_field(name="기본 명령어", value=(
        "`!scores <스포츠>` — 오늘 경기 결과\n"
        "`!standings <스포츠>` — 리그 순위\n"
        "`!leagues` — 전체 키워드 목록"
    ), inline=False)
    e.add_field(name="⚾ 분석 명령어", value=(
        "`!analyze mlb` — MLB 투수/타선 분석 + 승패 예측\n"
        "`!analyze kbo` — KBO 분석 (SPORTS_API_KEY 필요)\n"
        "`!analyze npb` — NPB 분석 (SPORTS_API_KEY 필요)\n"
        "`!odds` — MLB 언오버 배당 (ODDS_API_KEY 필요)"
    ), inline=False)
    e.add_field(name="API 키 설정 (cmd)", value=(
        "`set ODDS_API_KEY=키` → the-odds-api.com 무료\n"
        "`set SPORTS_API_KEY=키` → api-sports.io 무료"
    ), inline=False)
    return e

# ── 명령어 ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"✅ 로그인: {bot.user}  |  서버 수: {len(bot.guilds)}")
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching, name="⚾ 스포츠 분석"))

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

@bot.command(name="analyze", aliases=["분석"])
async def cmd_analyze(ctx, sport: str = "mlb"):
    sport = sport.lower()
    async with ctx.typing():
        try:
            if sport == "mlb" or sport == "야구":
                games = await get_mlb_schedule()
                odds_map = await get_odds_data()
                embed = await build_mlb_embed(games, odds_map)
                await ctx.send(embed=embed)
            elif sport in ASIA_LEAGUES:
                if not SPORTS_KEY:
                    await ctx.send(
                        f"❌ {ASIA_LEAGUES[sport]['name']} 분석을 사용하려면 SPORTS_API_KEY가 필요합니다.\n"
                        "무료 키 발급: https://api-sports.io\n"
                        "설정: `set SPORTS_API_KEY=발급받은키` (cmd에서 입력)"
                    )
                    return
                league_id = ASIA_LEAGUES[sport]["id"]
                games      = await get_asian_games(league_id)
                team_stats = await get_asian_team_stats(league_id)
                if games is None:
                    await ctx.send("❌ 데이터를 가져올 수 없습니다. API 키를 확인하세요.")
                    return
                embed = build_asian_embed(ASIA_LEAGUES[sport]["name"], games, team_stats or {})
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ `!analyze mlb`, `!analyze kbo`, `!analyze npb` 중 하나를 입력하세요.")
        except Exception as ex:
            await ctx.send(f"❌ 오류: {ex}")

@bot.command(name="odds", aliases=["배당", "언오버"])
async def cmd_odds(ctx):
    if not ODDS_KEY:
        await ctx.send(
            "❌ 언오버 배당을 사용하려면 ODDS_API_KEY가 필요합니다.\n"
            "무료 키 발급: **the-odds-api.com** (월 500회 무료)\n"
            "설정: `set ODDS_API_KEY=발급받은키` (cmd에서 입력 후 봇 재시작)"
        )
        return
    async with ctx.typing():
        try:
            odds_map = await get_odds_data()
            await ctx.send(embed=build_odds_embed(odds_map))
        except Exception as ex:
            await ctx.send(f"❌ 오류: {ex}")

@bot.command(name="debugleagues", aliases=["리그확인"])
async def cmd_debugleagues(ctx):
    """api-sports.io에서 사용 가능한 야구 리그 목록 확인"""
    if not SPORTS_KEY:
        await ctx.send("❌ SPORTS_API_KEY가 없습니다.")
        return
    async with ctx.typing():
        try:
            s = await http()
            headers = {"x-apisports-key": SPORTS_KEY}
            async with s.get(f"{SPORTS_API}/leagues", headers=headers) as r:
                data = await r.json()
            leagues = data.get("response", [])
            lines = []
            for lg in leagues:
                country = lg.get("country", {}).get("name", "?")
                cl = country.lower()
                if any(k in cl for k in ("korea", "japan", "united states", "usa")):
                    lid  = lg.get("id", "?")
                    name = lg.get("name", "?")
                    lines.append(f"`{lid}` {name} ({country})")
            e = discord.Embed(title="한국/일본/미국 야구 리그", color=0x3D6CFF,
                              description="\n".join(lines) or "없음")
            await ctx.send(embed=e)
        except Exception as ex:
            await ctx.send(f"❌ 오류: {ex}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ 인자 누락: `!help` 로 사용법을 확인하세요.")
    else:
        raise error

bot.run(TOKEN)
