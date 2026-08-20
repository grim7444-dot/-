from ..config import Config
from .base import BaseExchange


def create_exchange(config: Config) -> BaseExchange:
    name = config.exchange_name.lower()

    if name == "kiwoom":
        from .kiwoom_exchange import KiwoomExchange
        return KiwoomExchange(
            account_no=config.api_keys.get("account_no", ""),
            dry_run=config.dry_run,
            market=config.api_keys.get("market", "stock"),
        )

    from .ccxt_exchange import CCXTExchange
    return CCXTExchange(
        exchange_name=name,
        api_keys=config.api_keys,
        testnet=config.testnet,
        dry_run=config.dry_run,
    )
