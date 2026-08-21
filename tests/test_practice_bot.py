import pandas as pd

from trading.practice_bot import PracticeBotConfig, PracticeTradingBot


class FakeClient:
    def __init__(self):
        self.orders = []
        self.closes = []

    def get_open_positions(self):
        return pd.DataFrame()

    def place_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"ok": True}

    def close_position(self, *args, **kwargs):
        self.closes.append((args, kwargs))
        return {"ok": True}


class StubBot(PracticeTradingBot):
    def __init__(self, config, client, event):
        super().__init__(config, client=client)
        self.event = event

    def _load_signal(self, symbol, now):
        return pd.Timestamp("2026-08-21T12:00:00Z"), pd.Series(
            {
                "signal_event": self.event,
                "close": 1.1000,
                "stop_price": 1.0950,
                "take_profit_price": 1.1150,
            }
        )


def test_practice_bot_dry_run_does_not_submit_order():
    client = FakeClient()
    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",), dry_run=True),
        client,
        "long_entry",
    )

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime())

    assert client.orders == []


def test_practice_bot_submits_protected_practice_entry_when_requested():
    client = FakeClient()
    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",), dry_run=True, units_per_trade=2000),
        client,
        "long_entry",
    )

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime(), execute=True)

    assert client.orders == [
        {
            "symbol": "EUR_USD",
            "units": 2000,
            "stop_loss_price": 1.095,
            "take_profit_price": 1.115,
        }
    ]
