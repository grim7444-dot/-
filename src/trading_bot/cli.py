import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="암호화폐 자동 트레이딩 봇")
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--dry-run", action="store_true", help="시뮬레이션 모드 강제 활성화")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    from .config import Config
    from .bot import TradingBot
    import os

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    config = Config(args.config)
    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
