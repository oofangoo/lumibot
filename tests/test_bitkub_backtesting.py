"""BitkubBacktestingData and its use by the real backtest engine. No network: a scripted history stands in for Bitkub.

The engine tests compare a real ``Strategy.run_backtest`` against an independent pandas model on a price path built so
that
any lookahead (seeing a candle's close at its own start, or filling at the wrong candle's open) changes the trades.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from lumibot.backtesting import BitkubBacktesting
from lumibot.data_sources.bitkub_backtesting_data import BitkubBacktestingData
from lumibot.data_sources.exceptions import UnavailabeTimestep
from lumibot.entities import Asset, TradingFee
from lumibot.strategies import Strategy

BTC = Asset("BTC", Asset.AssetType.CRYPTO)
THB = Asset("THB", Asset.AssetType.FOREX)
TZ = "America/New_York"


def candles(n, start="2021-12-01", freq="D", base=1000.0, step=10.0):
    """Candle k starts at start + k*freq (UTC, like Bitkub): open = base+step*k, close = open+5."""
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="datetime")
    k = range(n)
    return pd.DataFrame(
        {
            "open": [base + step * i for i in k],
            "high": [base + step * i + 8 for i in k],
            "low": [base + step * i - 2 for i in k],
            "close": [base + step * i + 5 for i in k],
            "volume": [1.0] * n,
        },
        index=index,
    )


class FakeHistory:
    """Serves candles from memory the way BitkubHistory does: completed candles that fit inside [start, end]."""

    def __init__(self, frames):
        self.frames = frames  # {(symbol, resolution): DataFrame}
        self.calls = []

    def get(self, symbol, resolution, start_ts, end_ts):
        self.calls.append((symbol, resolution, start_ts, end_ts))
        frame = self.frames.get((symbol, resolution))
        if frame is None:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC")
            )
        step = {"1": 60, "5": 300, "15": 900, "60": 3600, "240": 14400, "1D": 86400}[resolution]
        start, end = pd.Timestamp(start_ts, unit="s", tz="UTC"), pd.Timestamp(end_ts, unit="s", tz="UTC")
        return frame[(frame.index >= start) & (frame.index + pd.Timedelta(seconds=step) <= end)]


def make_source(frames, start=datetime(2022, 1, 5), end=datetime(2022, 1, 12), **kwargs):
    history = FakeHistory(frames)
    source = BitkubBacktestingData(start, end, history=history, **kwargs)
    return source, history


def at(source, when):
    source._datetime = pd.Timestamp(when, tz=TZ)
    return source


# ------------------------------------------------------------------ symbols and timesteps
@pytest.mark.parametrize(
    "given, quote, symbol",
    [
        (BTC, None, "BTC_THB"),
        ("BTC", None, "BTC_THB"),
        ("btc/thb", None, "BTC_THB"),
        ("ETH_USDT", None, "ETH_USDT"),
        ((BTC, THB), None, "BTC_THB"),
        (BTC, Asset("USDT", Asset.AssetType.CRYPTO), "BTC_USDT"),
        ("BTC", "USDT", "BTC_USDT"),
    ],
)
def test_symbol_resolution(given, quote, symbol):
    source, _ = make_source({})
    base, resolved_quote = source._sanitize_base_and_quote_asset(given, quote)
    assert source.symbol_for(base, resolved_quote) == symbol


def test_non_crypto_and_malformed_symbols_are_refused():
    source, _ = make_source({})
    with pytest.raises(ValueError):
        source._sanitize_base_and_quote_asset(Asset("SPY", Asset.AssetType.STOCK))
    with pytest.raises(ValueError):
        source._sanitize_base_and_quote_asset("A_B_C")


@pytest.mark.parametrize(
    "timestep, resolution",
    [("minute", "1"), ("5 minutes", "5"), ("15 minutes", "15"), ("hour", "60"), ("4 hours", "240"), ("day", "1D")],
)
def test_timesteps_map_to_bitkub_resolutions(timestep, resolution):
    source, _ = make_source({})
    assert source._parse_source_timestep(timestep, reverse=True) == resolution


def test_unsupported_timestep_is_refused_not_substituted():
    source, _ = make_source({})
    with pytest.raises(UnavailabeTimestep):
        source._parse_source_timestep("30 minutes", reverse=True)


# ------------------------------------------------------------------ what a strategy can see
def test_only_completed_candles_are_visible_never_the_one_starting_now():
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    at(source, "2022-01-05 00:00")  # = 05:00 UTC: the candle that started 00:00 UTC today is still forming
    bars = source.get_historical_prices(BTC, 3, "day", quote=THB)
    assert bars.df.index[-1] == pd.Timestamp(
        "2022-01-04 00:00", tz="UTC"
    )  # yesterday's candle (k=34): finished at 00:00 UTC
    assert bars.df["close"].iloc[-1] == 1345.0 and len(bars.df) == 3
    assert source.get_last_price(BTC, quote=THB, timestep="day") == 1345.0


def test_a_candle_becomes_visible_the_moment_it_finishes_and_not_before():
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    source._datetime = pd.Timestamp("2022-01-05 23:59:59", tz="UTC").tz_convert(TZ)
    assert source.get_last_price(BTC, quote=THB, timestep="day") == 1345.0  # candle k=35 ends at 00:00 UTC on the 6th
    source._datetime = pd.Timestamp("2022-01-06 00:00:00", tz="UTC").tz_convert(TZ)
    assert source.get_last_price(BTC, quote=THB, timestep="day") == 1355.0


def test_a_negative_timeshift_reveals_the_current_candle_for_order_execution():
    """The backtesting broker asks for the candle that starts now by passing timeshift = -bar_duration."""
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    at(source, "2022-01-05 00:00")
    bars = source.get_historical_prices(BTC, 1, "day", quote=THB, timeshift=-timedelta(days=1))
    assert bars.df.index[-1] == pd.Timestamp("2022-01-05 00:00", tz="UTC") and bars.df["open"].iloc[-1] == 1350.0


def test_an_integer_timeshift_goes_back_that_many_bars():
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    at(source, "2022-01-05 00:00")
    assert source.get_historical_prices(BTC, 1, "day", quote=THB, timeshift=2).df["close"].iloc[-1] == 1325.0


def test_history_is_downloaded_once_for_the_whole_window_with_look_back():
    source, history = make_source({("BTC_THB", "1D"): candles(400, start="2021-01-01")})
    at(source, "2022-01-05 00:00")
    for _ in range(3):
        source.get_historical_prices(BTC, 5, "day", quote=THB)
        source.get_last_price(BTC, quote=THB, timestep="day")
    assert len(history.calls) == 1
    symbol, resolution, start_ts, end_ts = history.calls[0]
    assert (symbol, resolution) == ("BTC_THB", "1D")
    assert pd.Timestamp(start_ts, unit="s", tz=TZ) <= pd.Timestamp("2022-01-05", tz=TZ) - pd.Timedelta(
        days=300
    )  # warm-up data


def test_a_market_with_no_candles_gives_none_not_a_made_up_price():
    source, _ = make_source({})
    at(source, "2022-01-05 00:00")
    assert source.get_historical_prices(Asset("NOPE", Asset.AssetType.CRYPTO), 5, "day", quote=THB) is None
    assert source.get_last_price(Asset("NOPE", Asset.AssetType.CRYPTO), quote=THB, timestep="day") is None


def test_candles_that_never_traded_stay_absent_not_forward_filled():
    thin = candles(10 * 24 * 60, start="2022-01-01", freq="min", step=0.001).iloc[::37]  # a sparse minute series
    source, _ = make_source({("BTC_THB", "1"): thin}, start=datetime(2022, 1, 3), end=datetime(2022, 1, 4))
    at(source, "2022-01-03 12:00")
    frame = source.get_historical_prices(BTC, 20, "minute", quote=THB).df
    assert len(frame) == 20 and (frame.index.to_series().diff().dropna() == pd.Timedelta(minutes=37)).all()
    everything = source.get_historical_prices(BTC, 500, "minute", quote=THB).df
    assert 20 < len(everything) < 500  # asking for more than exists returns what exists, never padded


def test_between_dates_returns_completed_candles_in_the_window_for_the_benchmark():
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    bars = source.get_historical_prices_between_dates(
        BTC, "day", quote=THB, start_date=pd.Timestamp("2022-01-05", tz=TZ), end_date=pd.Timestamp("2022-01-09", tz=TZ)
    )
    assert bars.df.index[0] >= pd.Timestamp("2022-01-05", tz=TZ) and bars.df.index[-1] + pd.Timedelta(
        days=1
    ) <= pd.Timestamp("2022-01-09", tz=TZ)
    assert "return" in bars.df.columns


def test_quote_carries_only_the_last_close_and_invents_no_spread():
    source, _ = make_source({("BTC_THB", "1D"): candles(80)})
    at(source, "2022-01-05 00:00")
    source._timestep = "day"  # the engine sets this from the strategy's sleeptime
    quote = source.get_quote(BTC, quote=THB)
    assert quote.price == 1345.0 and quote.bid is None and quote.ask is None


def test_option_chains_do_not_exist():
    source, _ = make_source({})
    with pytest.raises(NotImplementedError):
        source.get_chains(BTC)


# ------------------------------------------------------------------ the real engine
FEES = dict(buy_trading_fees=[TradingFee(percent_fee=0.0025)], sell_trading_fees=[TradingFee(percent_fee=0.0025)])
QUIET = dict(
    show_plot=False,
    show_tearsheet=False,
    save_tearsheet=False,
    save_logfile=False,
    show_progress_bar=False,
    quiet_logs=True,
    show_indicators=False,
    save_stats_file=False,
)


def run(strategy_class, frames, start, end, analyze=False, **kwargs):
    return strategy_class.run_backtest(
        BitkubBacktesting,
        backtesting_start=start,
        backtesting_end=end,
        history=FakeHistory(frames),
        quote_asset=THB,
        budget=100_000,
        analyze_backtest=analyze,
        **{**QUIET, **kwargs},
    )


class Recorder(Strategy):
    """Buys 1 BTC on the first iteration and records exactly what it saw and what it got."""

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.seen, self.fills = [], []

    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 3, "day", quote=THB)
        self.seen.append(
            (self.get_datetime(), float(self.get_last_price(BTC, quote=THB)), float(bars.df["close"].iloc[-1]))
        )
        if len(self.seen) == 1:
            self.submit_order(self.create_order(BTC, 1, "buy", quote=THB))

    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.fills.append((self.get_datetime(), float(price), float(quantity)))


def test_engine_shows_only_finished_candles_and_fills_at_the_current_candles_open():
    _, strategy = run(
        Recorder, {("BTC_THB", "1D"): candles(80)}, datetime(2022, 1, 5), datetime(2022, 1, 12), benchmark_asset=None
    )
    # iteration at 00:00 NY on Jan 5 = 05:00 UTC: the newest finished candle began Jan 4 00:00 UTC (close 1345)
    assert [price for _, price, _ in strategy.seen] == [1345.0 + 10 * i for i in range(len(strategy.seen))]
    assert all(last == bar_close for _, last, bar_close in strategy.seen)
    # the order placed in that same iteration fills at the open of the candle that started Jan 5 00:00 UTC (1350),
    # not at 1355 (that candle's close)
    assert [(price, qty) for _, price, qty in strategy.fills] == [(1350.0, 1.0)]


def test_engine_charges_the_fee_and_values_the_position_at_the_last_finished_close():
    _, strategy = run(
        Recorder,
        {("BTC_THB", "1D"): candles(80)},
        datetime(2022, 1, 5),
        datetime(2022, 1, 12),
        benchmark_asset=None,
        **FEES,
    )
    assert strategy.cash == pytest.approx(100_000 - 1350.0 * 1.0025)  # 0.25% on the 1,350 THB purchase
    expected_value = strategy.cash + 1.0 * strategy.get_last_price(BTC, quote=THB)
    assert strategy.portfolio_value == pytest.approx(expected_value)


class SmaTrend(Strategy):
    """Long 1 BTC while the newest finished close is above the mean of the last 4 finished closes; flat otherwise."""

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.fills = []

    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 4, "day", quote=THB)
        if bars is None or len(bars.df) < 4:
            return
        closes = bars.df["close"]
        want_long = closes.iloc[-1] > closes.mean()
        position = self.get_position(BTC)
        held = float(position.quantity) if position else 0.0
        if want_long and held == 0:
            self.submit_order(self.create_order(BTC, 1, "buy", quote=THB))
        elif not want_long and held > 0:
            self.submit_order(self.create_order(BTC, held, "sell", quote=THB))

    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.fills.append((self.get_datetime().date(), str(order.side), float(price)))


def zigzag_frame():
    """Daily candles from 2021-12-01 with open[k] = close[k-1] and a path that turns several times."""
    closes = [
        100,
        101,
        102,
        103,
        104,
        106,
        109,
        113,
        118,
        117,
        115,
        112,
        108,
        103,
        99,
        96,
        95,
        97,
        101,
        106,
        112,
        119,
        125,
        124,
        122,
        119,
        115,
        110,
        104,
        100,
        98,
        99,
        103,
        108,
        114,
        121,
        128,
        133,
        131,
        127,
        122,
        116,
        110,
        105,
        103,
        104,
        108,
        114,
        121,
        129,
        138,
        146,
        150,
        148,
        143,
        137,
        130,
        124,
        119,
        117,
    ]
    closes = [float(c) * 10 for c in closes]
    opens = [closes[0]] + closes[:-1]
    index = pd.date_range("2021-12-01", periods=len(closes), freq="D", tz="UTC", name="datetime")
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": 1.0,
        },
        index=index,
    )


def independent_sma_model(frame, first_day, last_day):
    """A pandas re-implementation without LumiBot: decide on finished candles, fill at the current candle's open."""
    trades, held = [], False
    for day in pd.date_range(first_day, last_day, freq="D"):
        finished = frame[
            frame.index + pd.Timedelta(days=1) <= pd.Timestamp(day.date(), tz="UTC") + pd.Timedelta(hours=5)
        ]
        if len(finished) < 4:
            continue
        last4 = finished["close"].tail(4)
        want_long = last4.iloc[-1] > last4.mean()
        fill_open = frame.loc[pd.Timestamp(day.date(), tz="UTC"), "open"]  # the candle that starts at 00:00 UTC today
        if want_long and not held:
            trades.append((day.date(), "buy", fill_open))
            held = True
        elif not want_long and held:
            trades.append((day.date(), "sell", fill_open))
            held = False
    return trades


def test_engine_trades_match_an_independent_pandas_model_on_a_path_that_exposes_lookahead():
    frame = zigzag_frame()
    _, strategy = run(
        SmaTrend, {("BTC_THB", "1D"): frame}, datetime(2021, 12, 20), datetime(2022, 1, 28), benchmark_asset=None
    )
    expected = independent_sma_model(frame, "2021-12-20", "2022-01-27")
    actual = [(d, side, price) for d, side, price in strategy.fills]
    assert len(expected) >= 4  # the path really does produce several round trips
    assert actual == expected


def test_a_lookahead_version_of_the_rule_would_have_traded_differently():
    """Guards the guard: if the model's decisions are insensitive to lookahead, the parity test above proves nothing."""
    frame = zigzag_frame()
    honest = independent_sma_model(frame, "2021-12-20", "2022-01-27")
    peeking = []
    held = False
    for day in pd.date_range("2021-12-20", "2022-01-27", freq="D"):
        visible = frame[
            frame.index <= pd.Timestamp(day.date(), tz="UTC")
        ]  # includes the candle starting today (its close!)
        last4 = visible["close"].tail(4)
        want_long = last4.iloc[-1] > last4.mean()
        price = frame.loc[pd.Timestamp(day.date(), tz="UTC"), "open"]
        if want_long and not held:
            peeking.append((day.date(), "buy", price))
            held = True
        elif not want_long and held:
            peeking.append((day.date(), "sell", price))
            held = False
    assert peeking != honest


def test_benchmark_can_be_buy_and_hold_of_the_same_market():
    _, strategy = run(
        Recorder,
        {("BTC_THB", "1D"): candles(80)},
        datetime(2022, 1, 5),
        datetime(2022, 1, 12),
        benchmark_asset=(BTC, THB),
    )
    returns = strategy._benchmark_returns_df
    assert returns is not None and len(returns) >= 5
    assert returns["symbol_cumprod"].iloc[-1] == pytest.approx(
        returns["close"].iloc[-1] / returns["close"].iloc[0], rel=0.05
    )
