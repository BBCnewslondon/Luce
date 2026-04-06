import pandas as pd

import execution.mtf_oanda_bot as bot_module
from execution.mtf_oanda_bot import MTFOandaBotConfig, MTFOandaTradingBot


class _FakeOandaClient:
    def __init__(self, positions: pd.DataFrame):
        self._positions = positions
        self.closed = 0
        self.orders = []

    def get_open_positions(self) -> pd.DataFrame:
        return self._positions.copy()

    def close_position(self, symbol: str, long_units: str = "ALL", short_units: str = "ALL") -> dict:
        self.closed += 1
        return {"symbol": symbol, "closed": True, "long_units": long_units, "short_units": short_units}

    def place_market_order(self, symbol: str, units: int) -> dict:
        self.orders.append((symbol, units))
        return {"symbol": symbol, "units": units}


def _sample_mtf_data() -> dict:
    idx5 = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    idx4 = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    ohlcv5 = pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]}, index=idx5)
    ohlcv4 = pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]}, index=idx4)
    return {"5m": ohlcv5, "4h": ohlcv4}


def test_run_cycle_rebalances_when_signal_differs_from_current_position():
    signals = pd.DataFrame(
        {"signal_event": ["flat", "long_entry"], "position": [0, 1]},
        index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
    )
    client = _FakeOandaClient(positions=pd.DataFrame({"symbol": ["EUR_USD"], "net_units": [0]}))
    bot = MTFOandaTradingBot(
        config=MTFOandaBotConfig(symbol="EUR_USD", order_units=2500, dry_run=False),
        oanda_client=client,
        data_fetcher=lambda **_: _sample_mtf_data(),
        signal_builder=lambda **_: signals,
    )

    out = bot.run_cycle(now=pd.Timestamp("2024-02-01T00:00:00Z"))

    assert out["action"] == "rebalance_position"
    assert out["target_units"] == 2500
    assert client.orders == [("EUR_USD", 2500)]


def test_run_cycle_rebalances_to_short_when_signal_is_short():
    signals = pd.DataFrame(
        {"signal_event": ["flat", "short_entry"], "position": [0, -1]},
        index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
    )
    client = _FakeOandaClient(positions=pd.DataFrame({"symbol": ["EUR_USD"], "net_units": [0]}))
    bot = MTFOandaTradingBot(
        config=MTFOandaBotConfig(symbol="EUR_USD", order_units=1500, dry_run=False),
        oanda_client=client,
        data_fetcher=lambda **_: _sample_mtf_data(),
        signal_builder=lambda **_: signals,
    )

    out = bot.run_cycle(now=pd.Timestamp("2024-02-01T00:00:00Z"))

    assert out["action"] == "rebalance_position"
    assert out["target_units"] == -1500
    assert client.orders == [("EUR_USD", -1500)]


def test_run_cycle_closes_open_position_when_strategy_goes_flat():
    signals = pd.DataFrame(
        {"signal_event": ["long_entry", "filter_exit"], "position": [1, 0]},
        index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"),
    )
    client = _FakeOandaClient(positions=pd.DataFrame({"symbol": ["EUR_USD"], "net_units": [1000]}))
    bot = MTFOandaTradingBot(
        config=MTFOandaBotConfig(symbol="EUR_USD", order_units=1000, dry_run=False),
        oanda_client=client,
        data_fetcher=lambda **_: _sample_mtf_data(),
        signal_builder=lambda **_: signals,
    )

    out = bot.run_cycle(now=pd.Timestamp("2024-02-01T00:00:00Z"))

    assert out["action"] == "close_position"
    assert client.closed == 1


def test_run_from_env_parses_boolean_and_integer_values(monkeypatch):
    monkeypatch.setenv("BOT_SYMBOL", "EUR_USD")
    monkeypatch.setenv("BOT_ORDER_UNITS", "2000")
    monkeypatch.setenv("BOT_LOOKBACK_DAYS", "20")
    monkeypatch.setenv("BOT_DRY_RUN", "false")

    captured = {}

    class _DummyClient:
        pass

    class _DummyBot:
        def __init__(self, config, oanda_client):
            captured["config"] = config
            captured["client"] = oanda_client

        def run_cycle(self):
            return {"ok": True}

    monkeypatch.setattr(bot_module, "MTFOandaTradingBot", _DummyBot)
    monkeypatch.setattr(bot_module, "_get_oanda_client_cls", lambda: _DummyClient)

    out = bot_module.run_from_env()

    assert out == {"ok": True}
    assert captured["config"].symbol == "EUR_USD"
    assert captured["config"].order_units == 2000
    assert captured["config"].lookback_days == 20
    assert captured["config"].dry_run is False
