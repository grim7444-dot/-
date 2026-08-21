import asyncio
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
PREFIX = os.getenv("COMMAND_PREFIX", "!")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready():
    print(f"✅ 로그인 완료: {bot.user} (ID: {bot.user.id})")
    print(f"   서버 수: {len(bot.guilds)}")
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="⚽🏀⚾ 스포츠 경기",
        )
    )


@bot.command(name="help", aliases=["도움말"])
async def help_cmd(ctx: commands.Context):
    from utils.formatter import build_help_embed
    await ctx.send(embed=build_help_embed())


async def main():
    async with bot:
        await bot.load_extension("cogs.sports")
        await bot.start(TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
