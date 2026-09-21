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


def test_practice_bot_submits_protected_practice_entry_when_requested(tmp_path):
    client = FakeClient()
    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",), dry_run=True, units_per_trade=2000),
        client,
        "long_entry",
    )
    bot.log_dir = tmp_path
    bot.trades_csv = tmp_path / "trades.csv"
    bot._ensure_trades_csv()

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime(), execute=True)

    assert client.orders == [
        {
            "symbol": "EUR_USD",
            "units": 2000,
            "stop_loss_price": 1.095,
            "take_profit_price": 1.115,
        }
    ]
    assert bot.trades_csv.exists()
    content = bot.trades_csv.read_text()
    assert "EUR_USD,LONG,2000" in content


def test_practice_bot_submits_exit_when_in_position(tmp_path):
    client = FakeClient()
    client._positions = pd.DataFrame([{"symbol": "EUR_USD", "net_units": 1000}])
    client.get_open_positions = lambda: client._positions

    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",), dry_run=False),
        client,
        "stop_exit",
    )
    bot.log_dir = tmp_path
    bot.trades_csv = tmp_path / "trades.csv"
    bot._ensure_trades_csv()

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime(), execute=True)

    assert len(client.closes) == 1
    assert client.closes[0] == (("EUR_USD",), {"long_units": "ALL", "short_units": "NONE"})


def test_practice_bot_skips_when_max_positions_reached():
    client = FakeClient()
    # Already 2 positions open and max is 2
    client._positions = pd.DataFrame([
        {"symbol": "GBP_USD", "net_units": 1000},
        {"symbol": "USD_JPY", "net_units": 1000},
    ])
    client.get_open_positions = lambda: client._positions

    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",), max_open_positions=2),
        client,
        "long_entry",
    )

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime(), execute=True)
    assert client.orders == []


def test_practice_bot_skips_duplicate_candle():
    client = FakeClient()
    bot = StubBot(
        PracticeBotConfig(symbols=("EUR_USD",)),
        client,
        "long_entry",
    )

    bot.run_once(now=pd.Timestamp("2026-08-21T12:05:00Z").to_pydatetime(), execute=True)
    assert len(client.orders) == 1

    # Second run with same candle time should not submit again
    bot.run_once(now=pd.Timestamp("2026-08-21T12:06:00Z").to_pydatetime(), execute=True)
    assert len(client.orders) == 1


def test_practice_bot_rejects_invalid_protection_levels():
    client = FakeClient()
    bot = PracticeTradingBot(
        PracticeBotConfig(symbols=("EUR_USD",)),
        client=client,
    )
    # Long entry where stop >= close
    bad_long_signal = pd.Series({
        "signal_event": "long_entry",
        "close": 1.1000,
        "stop_price": 1.1050,  # invalid!
        "take_profit_price": 1.1150,
    })
    bot._submit_entry("EUR_USD", "long_entry", bad_long_signal, execute=True)
    assert client.orders == []

    # Short entry where target >= close
    bad_short_signal = pd.Series({
        "signal_event": "short_entry",
        "close": 1.1000,
        "stop_price": 1.1050,
        "take_profit_price": 1.1020,  # invalid!
    })
    bot._submit_entry("EUR_USD", "short_entry", bad_short_signal, execute=True)
    assert client.orders == []


def test_load_config():
    from trading.practice_bot import load_config
    cfg = load_config("config/settings.yaml")
    assert len(cfg.symbols) > 0
    assert cfg.poll_seconds > 0
    assert cfg.environment == "practice"


def test_practice_bot_run_once_full_pipeline(monkeypatch, tmp_path):
    from tests.test_mtf_forex_backtest import _make_ohlcv
    idx5 = pd.date_range("2024-01-01", periods=1200, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-25", periods=400, freq="4h", tz="UTC")
    m5 = _make_ohlcv(idx5, base=1.08, drift=0.00003)
    h4 = _make_ohlcv(idx4, base=1.06, drift=0.0004)

    monkeypatch.setattr(
        "trading.practice_bot.fetch_forex_data_oanda",
        lambda ticker, start, end, config: {"5m": m5, "4h": h4},
    )

    client = FakeClient()
    bot = PracticeTradingBot(
        PracticeBotConfig(symbols=("EUR_USD",), dry_run=True),
        client=client,
        log_dir=str(tmp_path),
    )

    bot.run_once(now=pd.Timestamp("2024-01-05T12:00:00Z").to_pydatetime())
    assert bot.trades_csv.exists()
    assert "EUR_USD" in bot._last_processed_candle


