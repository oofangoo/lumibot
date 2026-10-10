.. _backtesting.bitkub:

Bitkub Backtesting
==================

.. meta::
   :description: Backtest crypto strategies in LumiBot on real Bitkub THB and USDT candles, with no API key. Covers setup, fees, how candles and fills are timed, caching, and limits.

``BitkubBacktesting`` backtests spot crypto strategies on **real Bitkub candles** (THB and USDT markets). It needs **no API key** and no account: the history comes from Bitkub's public endpoints and is cached on your disk after the first download.

Use it when you plan to trade on Bitkub (see the :doc:`Bitkub broker guide <brokers.bitkub>`) and want to test the strategy in the same market and currency you will trade, including Bitkub's own fees and prices.

A Complete Example
------------------

.. code-block:: python

    from datetime import datetime

    from lumibot.backtesting import BitkubBacktesting
    from lumibot.entities import Asset, TradingFee
    from lumibot.strategies import Strategy

    BTC = Asset("BTC", Asset.AssetType.CRYPTO)
    THB = Asset("THB", Asset.AssetType.FOREX)


    class BtcTrend(Strategy):
        """Hold BTC while the last finished daily close is above its 50-day average, otherwise hold THB."""

        def initialize(self):
            self.sleeptime = "1D"
            self.set_market("24/7")

        def on_trading_iteration(self):
            bars = self.get_historical_prices(BTC, 50, "day", quote=THB)
            if bars is None or len(bars.df) < 50:
                return
            closes = bars.df["close"]
            held = self.get_position(BTC)
            if closes.iloc[-1] > closes.mean() and held is None:
                budget = self.get_cash() * 0.99
                price = self.get_last_price(BTC, quote=THB)
                self.submit_order(self.create_order(BTC, budget / price, "buy", quote=THB))
            elif closes.iloc[-1] <= closes.mean() and held is not None:
                self.submit_order(self.create_order(BTC, held.quantity, "sell", quote=THB))


    BtcTrend.backtest(
        BitkubBacktesting,
        backtesting_start=datetime(2021, 1, 1),
        backtesting_end=datetime(2025, 12, 31),
        quote_asset=THB,
        benchmark_asset=(BTC, THB),  # buy-and-hold BTC/THB as the benchmark
        budget=100_000,  # THB
        buy_trading_fees=[TradingFee(percent_fee=0.0025)],
        sell_trading_fees=[TradingFee(percent_fee=0.0025)],
    )

Fees and Costs
--------------

Pass fees to ``backtest()`` yourself. Bitkub's standard fee is **0.25% per side** (the same for maker and taker); check your own account tier. The candles carry no bid/ask, so there is **no spread or slippage** unless you add it with ``buy_trading_slippages`` / ``sell_trading_slippages``. THB cash earns no interest in the simulation.

How Candles and Fills Are Timed
-------------------------------

Bitkub stamps each candle with its **start** time. The source is built so a backtest cannot use information from the future:

- **Strategies only see finished candles.** A candle becomes visible when it has closed. ``get_historical_prices`` and ``get_last_price`` never include the candle that is still forming, so a day's close is not available at the start of that day.
- **Orders fill on the candle that starts now.** Market orders fill at its open, limit orders are checked against its range. In practice a daily strategy decides from yesterday's finished close and trades at today's open, exactly the price at the boundary where the decision became known.
- **Daily strategies run at the start of each simulated day** in LumiBot's default time zone, while Bitkub's daily candles start at 00:00 UTC. At each decision the newest visible candle is the one that closed at 00:00 UTC that day.

.. warning::

    Do not feed Bitkub candles to ``PandasData`` yourself with their start timestamps. LumiBot would then show a strategy each candle's full close at the instant it opens. Use ``BitkubBacktesting``, which handles this.

What Is Never Done to the Data
------------------------------

- **No filling in.** Bitkub only has candles for periods that had trades, so minute candles for illiquid coins are sparse. Gaps stay gaps, and a market with no candles for your window returns ``None`` (with an error in the log) instead of an invented price.
- **Conflicting duplicates are resolved, not guessed.** The exchange sometimes returns two versions of the same candle with different volume, high and close. The most complete one (largest volume) is kept; corrupt rows (non-positive prices) are dropped.
- **Only completed candles are cached.** If a download fails part-way, nothing is recorded as downloaded, so a half-finished cache can never pass for full history.

Timesteps, History Length and Caching
-------------------------------------

Supported bar sizes are ``minute``, ``5 minutes``, ``15 minutes``, ``hour``, ``4 hours`` and ``day``; anything else raises ``UnavailabeTimestep`` rather than being approximated.

History is downloaded the first time a market and timestep are requested, for your whole backtest window plus 300 bars of look-back, and cached under LumiBot's cache folder (``<cache folder>/bitkub``; set ``LUMIBOT_CACHE_FOLDER`` to move it, or pass ``cache_dir=...``). Later runs reuse the cache and download only what is missing; the newest cached candles are re-checked each time the cache is extended. Bitkub's history for a market starts when that market was listed (BTC/THB from May 2018).

Limits
------

- Bitkub has **no stop orders**, so a stop-loss in a backtest is not something you can have on the exchange. Test the same logic your live strategy will actually run.
- Market orders fill at a candle's open with no order-book depth, so very large orders are optimistic.
- ``get_quote`` returns only the last finished close (no bid/ask).
- Spot only: no options, futures or leverage.
