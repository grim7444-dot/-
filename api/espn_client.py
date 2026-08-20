import aiohttp
from datetime import datetime, date
from config import ESPN_BASE_URL, SPORTS


def _resolve_sport_and_league(sport_input: str) -> tuple[str, str] | None:
    """Return (espn_sport_key, league_id) from a user-supplied sport name."""
    sport_input = sport_input.lower().strip()
    for sport_key, sport_data in SPORTS.items():
        if sport_input in sport_data["aliases"] or sport_input == sport_key:
            default = sport_data["default_league"]
            return sport_key, sport_data["leagues"][default]["id"]
    return None


def _resolve_league(sport_key: str, league_input: str) -> str | None:
    """Return ESPN league id for a given league alias."""
    leagues = SPORTS[sport_key]["leagues"]
    league_input = league_input.lower().strip()
    for key, data in leagues.items():
        if league_input == key or league_input in data["name"].lower():
            return data["id"]
    return None


class ESPNClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def _get(self, url: str, params: dict = None) -> dict:
        async with self.session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_scores(
        self,
        sport: str,
        league_id: str,
        target_date: date | None = None,
    ) -> list[dict]:
        url = f"{ESPN_BASE_URL}/{sport}/{league_id}/scoreboard"
        params = {}
        if target_date:
            params["dates"] = target_date.strftime("%Y%m%d")
        data = await self._get(url, params)
        return _parse_scoreboard(data)

    async def get_standings(self, sport: str, league_id: str) -> list[dict]:
        url = f"{ESPN_BASE_URL}/{sport}/{league_id}/standings"
        data = await self._get(url)
        return _parse_standings(data)

    async def get_team_record(self, sport: str, league_id: str, team_query: str) -> dict | None:
        url = f"{ESPN_BASE_URL}/{sport}/{league_id}/teams"
        data = await self._get(url)
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        for entry in teams:
            team = entry.get("team", {})
            name = team.get("displayName", "")
            short = team.get("abbreviation", "")
            if team_query.lower() in name.lower() or team_query.upper() == short.upper():
                records = team.get("record", {}).get("items", [])
                return _parse_team_record(team, records)
        return None


# ── parsers ──────────────────────────────────────────────────────────────────

def _parse_scoreboard(data: dict) -> list[dict]:
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {})
        competitors = comp.get("competitors", [])
        state = status.get("type", {}).get("state", "pre")   # pre / in / post
        detail = status.get("type", {}).get("shortDetail", "")

        teams = []
        for c in competitors:
            teams.append({
                "name":  c.get("team", {}).get("displayName", "?"),
                "abbr":  c.get("team", {}).get("abbreviation", "?"),
                "score": c.get("score", "-"),
                "winner": c.get("winner", False),
                "logo":  c.get("team", {}).get("logo", ""),
            })

        games.append({
            "name":   event.get("name", ""),
            "date":   event.get("date", ""),
            "state":  state,
            "detail": detail,
            "teams":  teams,
        })
    return games


def _parse_standings(data: dict) -> list[dict]:
    rows = []
    children = data.get("children", [])
    # Some leagues nest standings under children[].standings
    for child in children:
        entries = child.get("standings", {}).get("entries", [])
        if not entries:
            entries = child.get("entries", [])
        for entry in entries:
            team = entry.get("team", {})
            stats = {s["name"]: s.get("displayValue", s.get("value", "")) for s in entry.get("stats", [])}
            rows.append({
                "team": team.get("displayName", "?"),
                "abbr": team.get("abbreviation", "?"),
                "wins": stats.get("wins", stats.get("gamesWon", "-")),
                "losses": stats.get("losses", stats.get("gamesLost", "-")),
                "pct": stats.get("winPercent", stats.get("pointsPercentage", "-")),
                "gamesBack": stats.get("gamesBehind", ""),
                "extra": stats,
            })
    return rows


def _parse_team_record(team: dict, records: list) -> dict:
    overall = {}
    for item in records:
        if item.get("type") in ("total", "overall", None):
            for s in item.get("stats", []):
                overall[s["name"]] = s.get("displayValue", s.get("value", "-"))
            break
    if not overall and records:
        for s in records[0].get("stats", []):
            overall[s["name"]] = s.get("displayValue", s.get("value", "-"))

    return {
        "name":   team.get("displayName", "?"),
        "abbr":   team.get("abbreviation", "?"),
        "logo":   team.get("logos", [{}])[0].get("href", "") if team.get("logos") else team.get("logo", ""),
        "wins":   overall.get("wins", overall.get("gamesWon", "-")),
        "losses": overall.get("losses", overall.get("gamesLost", "-")),
        "pct":    overall.get("winPercent", "-"),
        "record": overall,
    }
