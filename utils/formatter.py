from datetime import datetime, timezone
import discord
from config import COLORS, SPORTS


def _sport_emoji(sport_key: str) -> str:
    return SPORTS.get(sport_key, {}).get("emoji", "🏅")


def _format_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%m/%d %H:%M UTC")
    except Exception:
        return iso


def build_scores_embed(sport_key: str, league_name: str, games: list[dict]) -> discord.Embed:
    emoji = _sport_emoji(sport_key)
    embed = discord.Embed(
        title=f"{emoji} {league_name} 경기 결과",
        color=COLORS["default"],
        timestamp=datetime.now(timezone.utc),
    )

    if not games:
        embed.description = "오늘 예정된 경기가 없습니다."
        return embed

    for g in games:
        teams = g["teams"]
        if len(teams) < 2:
            continue

        home = teams[0]
        away = teams[1]
        state = g["state"]
        detail = g["detail"]

        if state == "post":
            score_line = f"**{away['abbr']} {away['score']}** vs **{home['score']} {home['abbr']}**"
            winner = away if away["winner"] else home if home["winner"] else None
            win_tag = f"  ✅ {winner['name']} 승리" if winner else "  ⚖️ 무승부"
            value = f"{score_line}\n{win_tag}\n🕐 {_format_date(g['date'])}"
            embed.add_field(name="✔️ 종료", value=value, inline=False)
        elif state == "in":
            score_line = f"**{away['abbr']} {away['score']}** vs **{home['score']} {home['abbr']}**"
            value = f"{score_line}\n🔴 진행중 — {detail}"
            embed.add_field(name="▶️ 진행중", value=value, inline=False)
        else:
            value = f"{away['name']} vs {home['name']}\n⏰ {_format_date(g['date'])}"
            embed.add_field(name="📅 예정", value=value, inline=False)

    embed.set_footer(text="출처: ESPN")
    return embed


def build_standings_embed(sport_key: str, league_name: str, rows: list[dict]) -> discord.Embed:
    emoji = _sport_emoji(sport_key)
    embed = discord.Embed(
        title=f"{emoji} {league_name} 순위표",
        color=COLORS["default"],
        timestamp=datetime.now(timezone.utc),
    )

    if not rows:
        embed.description = "순위 데이터를 가져올 수 없습니다."
        return embed

    lines = []
    for i, row in enumerate(rows[:20], 1):
        pct = row.get("pct", "-")
        try:
            pct_str = f"{float(pct):.3f}"
        except Exception:
            pct_str = str(pct)
        gb = f"  GB:{row['gamesBack']}" if row.get("gamesBack") else ""
        lines.append(f"`{i:>2}.` **{row['abbr']:<4}** {row['wins']}승 {row['losses']}패  ({pct_str}){gb}")

    embed.description = "\n".join(lines)
    embed.set_footer(text="출처: ESPN")
    return embed


def build_team_embed(sport_key: str, league_name: str, record: dict) -> discord.Embed:
    emoji = _sport_emoji(sport_key)
    wins = record.get("wins", "-")
    losses = record.get("losses", "-")

    try:
        w, l = int(wins), int(losses)
        color = COLORS["win"] if w > l else COLORS["loss"] if l > w else COLORS["default"]
    except Exception:
        color = COLORS["default"]

    pct = record.get("pct", "-")
    try:
        pct_str = f"{float(pct):.3f}"
    except Exception:
        pct_str = str(pct)

    embed = discord.Embed(
        title=f"{emoji} {record['name']} ({record['abbr']})",
        description=f"**{league_name}**",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="승", value=str(wins), inline=True)
    embed.add_field(name="패", value=str(losses), inline=True)
    embed.add_field(name="승률", value=pct_str, inline=True)

    if record.get("logo"):
        embed.set_thumbnail(url=record["logo"])

    for key, val in record.get("record", {}).items():
        if key not in ("wins", "losses", "winPercent", "gamesWon", "gamesLost", "pointsPercentage"):
            embed.add_field(name=key, value=str(val), inline=True)

    embed.set_footer(text="출처: ESPN")
    return embed


def build_help_embed() -> discord.Embed:
    embed = discord.Embed(
        title="⚽🏀⚾🏈🏒 스포츠 점수 봇",
        description="각종 스포츠 승패 정보를 알려드립니다.",
        color=COLORS["default"],
    )
    embed.add_field(
        name="📋 명령어",
        value=(
            "`!scores <스포츠> [리그]` — 오늘 경기 결과\n"
            "`!standings <스포츠> [리그]` — 리그 순위\n"
            "`!team <스포츠> [리그] <팀이름>` — 팀 승패 기록\n"
            "`!sports` — 지원 스포츠 목록"
        ),
        inline=False,
    )
    embed.add_field(
        name="🌐 예시",
        value=(
            "`!scores soccer premier`\n"
            "`!scores 야구`\n"
            "`!standings basketball nba`\n"
            "`!team 축구 kleague 전북`\n"
            "`!team baseball mlb Yankees`"
        ),
        inline=False,
    )
    return embed


def build_sports_list_embed() -> discord.Embed:
    from config import SPORTS
    embed = discord.Embed(
        title="지원 스포츠 목록",
        color=COLORS["default"],
    )
    for sport_key, data in SPORTS.items():
        leagues = "\n".join(f"  • `{k}` — {v['name']}" for k, v in data["leagues"].items())
        aliases = ", ".join(f"`{a}`" for a in data["aliases"])
        embed.add_field(
            name=f"{data['emoji']} {sport_key}  (별칭: {aliases})",
            value=leagues,
            inline=False,
        )
    return embed
