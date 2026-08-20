from ..config import Config
from .base import BaseExchange
from .ccxt_exchange import CCXTExchange


def create_exchange(config: Config) -> BaseExchange:
    return CCXTExchange(
        exchange_name=config.exchange_name,
        api_keys=config.api_keys,
        testnet=config.testnet,
        dry_run=config.dry_run,
    )
