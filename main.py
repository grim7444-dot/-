#!/usr/bin/env python3
"""트레이딩 봇 진입점"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.cli import main

if __name__ == "__main__":
    main()
