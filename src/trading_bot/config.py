import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.exchange_name: str = data["exchange"]["name"]
        self.testnet: bool = data["exchange"].get("testnet", True)

        self.symbol: str = data["trading"]["symbol"]
        self.timeframe: str = data["trading"]["timeframe"]
        self.trade_amount: float = data["trading"]["trade_amount"]
        self.max_positions: int = data["trading"]["max_positions"]

        self.strategy_name: str = data["strategy"]["name"]
        self.strategy_params: dict = data["strategy"].get("params", {})

        self.stop_loss_pct: float = data["risk"]["stop_loss_pct"]
        self.take_profit_pct: float = data["risk"]["take_profit_pct"]
        self.max_drawdown_pct: float = data["risk"]["max_drawdown_pct"]

        self.log_level: str = data["logging"]["level"]
        self.log_file: str = data["logging"]["file"]

        self.dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"

        self._api_keys = self._load_api_keys()

    def _load_api_keys(self) -> dict:
        name = self.exchange_name.lower()
        if name == "binance":
            return {
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_SECRET_KEY", ""),
            }
        elif name == "upbit":
            return {
                "apiKey": os.getenv("UPBIT_ACCESS_KEY", ""),
                "secret": os.getenv("UPBIT_SECRET_KEY", ""),
            }
        elif name == "kiwoom":
            return {
                "account_no": os.getenv("KIWOOM_ACCOUNT_NO", ""),
                "market": os.getenv("KIWOOM_MARKET", "stock"),  # stock | futures
            }
        return {}

    @property
    def api_keys(self) -> dict:
        return self._api_keys
