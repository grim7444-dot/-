import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="암호화폐 자동 트레이딩 봇")
    sub = parser.add_subparsers(dest="command")

    # 실거래/DRY-RUN 서브커맨드
    run_p = sub.add_parser("run", help="실거래 또는 DRY-RUN 모드 실행")
    run_p.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    run_p.add_argument("--dry-run", action="store_true", help="시뮬레이션 강제 활성화")

    # 백테스트 서브커맨드
    bt_p = sub.add_parser("backtest", help="모의 데이터로 전략 백테스트")
    bt_p.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    bt_p.add_argument("--ticks", type=int, default=50, help="시뮬레이션 틱 수 (기본: 50)")
    bt_p.add_argument("--speed", type=float, default=0.1, help="틱 간격 초 (기본: 0.1)")
    bt_p.add_argument("--strategy", choices=["ma_crossover", "rsi", "macd"], help="전략 덮어쓰기")

    args = parser.parse_args()

    # 기본 커맨드가 없으면 backtest 실행
    if args.command is None:
        args.command = "backtest"
        args.config = "config/config.yaml"
        args.ticks = 50
        args.speed = 0.1
        args.strategy = None

    if not Path(args.config).exists():
        print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    from .config import Config
    import os

    config = Config(args.config)

    if args.command == "run":
        if args.dry_run:
            os.environ["DRY_RUN"] = "true"
        from .bot import TradingBot
        bot = TradingBot(config)
        bot.start()

    elif args.command == "backtest":
        if hasattr(args, "strategy") and args.strategy:
            config.strategy_name = args.strategy
        from .backtest import BacktestRunner
        runner = BacktestRunner(config, ticks=args.ticks, speed=args.speed)
        runner.run()


if __name__ == "__main__":
    main()
