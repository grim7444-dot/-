import aiohttp
import discord
from discord.ext import commands
from datetime import date, datetime

from config import SPORTS
from api.espn_client import ESPNClient, _resolve_sport_and_league, _resolve_league
from utils.formatter import (
    build_scores_embed,
    build_standings_embed,
    build_team_embed,
    build_help_embed,
    build_sports_list_embed,
)


def _lookup(sport_input: str) -> tuple[str, str, str] | None:
    """Return (sport_key, espn_sport, default_league_id) or None."""
    sport_input = sport_input.lower().strip()
    for key, data in SPORTS.items():
        if sport_input == key or sport_input in data["aliases"]:
            default = data["default_league"]
            return key, data["espn_sport"], data["leagues"][default]["id"]
    return None


def _league_id(sport_key: str, league_input: str) -> tuple[str, str] | None:
    """Return (league_id, league_name) given sport key and user league input."""
    leagues = SPORTS[sport_key]["leagues"]
    league_input = league_input.lower().strip()
    for k, v in leagues.items():
        if league_input == k or league_input in v["name"].lower():
            return v["id"], v["name"]
    return None


class Sports(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _client(self) -> ESPNClient:
        if not hasattr(self.bot, "_http_session") or self.bot._http_session.closed:
            self.bot._http_session = aiohttp.ClientSession()
        return ESPNClient(self.bot._http_session)

    @commands.command(name="help_sports", aliases=["스포츠도움말"])
    async def help_sports(self, ctx: commands.Context):
        await ctx.send(embed=build_help_embed())

    @commands.command(name="sports", aliases=["스포츠목록", "sportlist"])
    async def sports_list(self, ctx: commands.Context):
        await ctx.send(embed=build_sports_list_embed())

    @commands.command(name="scores", aliases=["경기결과", "score", "스코어"])
    async def scores(self, ctx: commands.Context, sport_input: str = "soccer", league_input: str = None):
        """오늘의 경기 결과: !scores <스포츠> [리그]"""
        res = _lookup(sport_input)
        if res is None:
            await ctx.send(f"❌ 알 수 없는 스포츠: `{sport_input}`\n`!sports` 명령어로 목록을 확인하세요.")
            return

        sport_key, espn_sport, league_id = res
        league_name = SPORTS[sport_key]["leagues"][SPORTS[sport_key]["default_league"]]["name"]

        if league_input:
            result = _league_id(sport_key, league_input)
            if result is None:
                await ctx.send(f"❌ 알 수 없는 리그: `{league_input}`\n`!sports` 명령어로 목록을 확인하세요.")
                return
            league_id, league_name = result

        async with ctx.typing():
            try:
                client = await self._client()
                games = await client.get_scores(espn_sport, league_id, date.today())
                embed = build_scores_embed(sport_key, league_name, games)
                await ctx.send(embed=embed)
            except aiohttp.ClientResponseError as e:
                await ctx.send(f"❌ API 오류 ({e.status}): 리그 정보를 가져올 수 없습니다.")
            except Exception as e:
                await ctx.send(f"❌ 오류 발생: {e}")

    @commands.command(name="standings", aliases=["순위", "standing"])
    async def standings(self, ctx: commands.Context, sport_input: str = "soccer", league_input: str = None):
        """리그 순위: !standings <스포츠> [리그]"""
        res = _lookup(sport_input)
        if res is None:
            await ctx.send(f"❌ 알 수 없는 스포츠: `{sport_input}`\n`!sports` 명령어로 목록을 확인하세요.")
            return

        sport_key, espn_sport, league_id = res
        league_name = SPORTS[sport_key]["leagues"][SPORTS[sport_key]["default_league"]]["name"]

        if league_input:
            result = _league_id(sport_key, league_input)
            if result is None:
                await ctx.send(f"❌ 알 수 없는 리그: `{league_input}`\n`!sports` 명령어로 목록을 확인하세요.")
                return
            league_id, league_name = result

        async with ctx.typing():
            try:
                client = await self._client()
                rows = await client.get_standings(espn_sport, league_id)
                embed = build_standings_embed(sport_key, league_name, rows)
                await ctx.send(embed=embed)
            except aiohttp.ClientResponseError as e:
                await ctx.send(f"❌ API 오류 ({e.status}): 순위 정보를 가져올 수 없습니다.")
            except Exception as e:
                await ctx.send(f"❌ 오류 발생: {e}")

    @commands.command(name="team", aliases=["팀", "팀기록"])
    async def team(self, ctx: commands.Context, sport_input: str = None, *args):
        """팀 승패 기록: !team <스포츠> [리그] <팀이름>"""
        if sport_input is None or not args:
            await ctx.send("사용법: `!team <스포츠> [리그] <팀이름>`\n예: `!team baseball mlb Yankees`")
            return

        res = _lookup(sport_input)
        if res is None:
            await ctx.send(f"❌ 알 수 없는 스포츠: `{sport_input}`")
            return

        sport_key, espn_sport, league_id = res
        league_name = SPORTS[sport_key]["leagues"][SPORTS[sport_key]["default_league"]]["name"]

        # Second arg might be a league override; try it first
        team_query = " ".join(args)
        if len(args) >= 2:
            result = _league_id(sport_key, args[0])
            if result:
                league_id, league_name = result
                team_query = " ".join(args[1:])

        async with ctx.typing():
            try:
                client = await self._client()
                record = await client.get_team_record(espn_sport, league_id, team_query)
                if record is None:
                    await ctx.send(f"❌ `{team_query}` 팀을 찾을 수 없습니다.")
                    return
                embed = build_team_embed(sport_key, league_name, record)
                await ctx.send(embed=embed)
            except aiohttp.ClientResponseError as e:
                await ctx.send(f"❌ API 오류 ({e.status}): 팀 정보를 가져올 수 없습니다.")
            except Exception as e:
                await ctx.send(f"❌ 오류 발생: {e}")

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ 필수 인자 누락: `{error.param.name}`\n`!help_sports` 로 도움말을 확인하세요.")
        elif isinstance(error, commands.CommandNotFound):
            pass
        else:
            raise error


async def setup(bot: commands.Bot):
    await bot.add_cog(Sports(bot))
