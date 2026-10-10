Bitkub Spot Crypto Trading with LumiBot
=======================================

.. meta::
   :description: Trade Thai baht (THB) and USDT spot crypto markets on Bitkub with LumiBot. Covers API keys, credentials, market data, order semantics, supported order types, and safety notes.

How to Use Bitkub
-----------------

`Bitkub <https://www.bitkub.com/>`_ is a Thai spot crypto exchange. The LumiBot integration can pull market data (no account needed) and, with an API key, read balances and trade **spot** markets quoted in **THB** (default) or **USDT**.

.. warning::

    Bitkub is a **real-money exchange with no paper-trading mode**. Every order a strategy submits is a live order. Start with a read-only API key, then a very small order, and keep a close eye on the first runs.

.. list-table:: Bitkub support at a glance
   :header-rows: 1
   :widths: 30 20 50

   * - Capability
     - Status
     - Notes
   * - Market data (last price, bid/ask, order book, OHLCV candles)
     - Supported
     - Public endpoints, **no API key required**
   * - Market and limit orders
     - Supported
     - On active ``exchange`` markets (see below)
   * - Cancel orders, balances, positions
     - Supported
     - Requires an API key
   * - Stop, stop-limit, trailing-stop orders
     - **Not supported**
     - Bitkub has no broker-side stop orders. Handle protective stops inside the strategy.
   * - Bracket, OCO, OTO, multi-leg orders
     - Not supported
     - Rejected with an explanatory error
   * - Modify an order
     - Not supported
     - Cancel and submit a new order
   * - Shorting, margin, futures, options
     - Not supported
     - Spot only
   * - Live order updates
     - Polling
     - REST polling every 5 seconds (WebSocket streaming is not implemented)
   * - Backtesting data source
     - Not included
     - ``BitkubData`` is a live data source

Account and API Key
-------------------

1. Create and verify a Bitkub account, then create an API key at `bitkub.com/en/api-management <https://www.bitkub.com/en/api-management>`_.
2. Give the key only the permissions you need. **Leave Deposit and Withdraw unchecked.** For read-only experiments, leave Trade unchecked too.
3. If you restrict the key to specific IP addresses, add the address of the machine that runs your strategy.

Never hardcode keys in strategy files; use environment variables or your secret manager.

Environment Variables
---------------------

.. code-block:: shell

    BITKUB_API_KEY=your_bitkub_api_key
    BITKUB_API_SECRET=your_bitkub_api_secret

    # Optional
    BITKUB_QUOTE_ASSET=THB          # default quote currency: THB or USDT
    BITKUB_SYMBOLS=BTC_THB,ETH_THB  # extra markets to poll for open orders (see "Open orders" below)

When ``BITKUB_API_KEY`` and ``BITKUB_API_SECRET`` are set and no other broker is configured, LumiBot selects Bitkub automatically. You can also force it with ``BROKER=bitkub``.

A Minimal Strategy
------------------

.. code-block:: python

    from lumibot.brokers import Bitkub
    from lumibot.credentials import BITKUB_CONFIG
    from lumibot.entities import Asset
    from lumibot.strategies import Strategy


    class BitkubBuyAndLog(Strategy):
        def initialize(self):
            self.sleeptime = "1M"
            self.set_market("24/7")

        def on_trading_iteration(self):
            btc = Asset("BTC", Asset.AssetType.CRYPTO)
            quote = self.quote_asset  # THB
            price = self.get_last_price(btc, quote=quote)
            self.log_message(f"BTC/THB last price: {price}")

            if self.first_iteration:
                # Buys roughly 0.0001 BTC; Bitkub's minimum order value is 10 THB.
                order = self.create_order(btc, 0.0001, "buy", order_type="market", quote=quote)
                self.submit_order(order)


    if __name__ == "__main__":
        broker = Bitkub(BITKUB_CONFIG)
        strategy = BitkubBuyAndLog(broker=broker, quote_asset=Asset("THB", Asset.AssetType.FOREX))
        strategy.run_live()

Crypto orders always need a ``quote`` (``Asset("THB", Asset.AssetType.FOREX)`` for THB markets, or a crypto ``USDT`` asset for USDT markets).

Market Data Only (No API Key)
-----------------------------

.. code-block:: python

    from lumibot.data_sources import BitkubData
    from lumibot.entities import Asset

    data = BitkubData()  # no credentials needed
    btc = Asset("BTC", Asset.AssetType.CRYPTO)

    data.get_last_price(btc)                         # THB by default
    quote = data.get_quote(btc)                      # bid, ask, last, 24h change
    daily = data.get_historical_prices(btc, 30, "day")
    book = data.get_orderbook(btc, depth=5)          # {"bids": [(price, size), ...], "asks": [...]}
    markets = data.get_symbols()                     # precision, minimum size, status, source

Candle behaviour worth knowing:

- Native bar sizes are ``minute``, ``5 minutes``, ``15 minutes``, ``hour``, ``4 hours`` and ``day``. Other sizes raise ``ValueError`` instead of being silently substituted.
- ``get_historical_prices`` returns **completed bars only**; use ``get_last_price`` or ``get_quote`` for the live price.
- Candles exist only for periods that had trades, so illiquid markets return **sparse** minute bars. Gaps are never filled in. If fewer bars than requested exist, you get what exists (a warning is logged); if none exist you get ``None``.
- The exchange sometimes returns several conflicting rows for one timestamp. LumiBot keeps the most complete version (largest volume).

Order Semantics
---------------

Bitkub sizes orders differently from LumiBot, so the broker converts:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Order
     - What Bitkub receives
     - Consequence
   * - **Buy**, limit
     - THB to spend = ``quantity × limit price`` (rounded down to the cent)
     - You buy ``quantity`` coins at the limit price or better
   * - **Buy**, market
     - THB to spend = ``quantity × current best ask``
     - You receive **approximately** ``quantity`` coins (less the fee and any order-book slippage)
   * - **Sell**, limit / market
     - The coin quantity (rounded down to the market's precision)
     - Exact quantity

Prices are rounded down to the market's price tick, and an order below the market's minimum value (``min_quote_size``, 10 THB on most markets) is refused before it is sent. The quantity LumiBot tracks for a buy is the executable quantity after rounding, and it is updated to the real filled quantity once the order completes.

Only markets with ``status == "active"`` and ``source == "exchange"`` can be traded through the API. Broker-sourced markets (for example some USDT pairs) are rejected by Bitkub's trading endpoints, and LumiBot reports that clearly instead of sending the order.

Cash and Positions
------------------

- **Cash** is the *total* balance of the quote currency, including THB reserved by open buy orders. A buy larger than the unreserved balance is rejected by Bitkub and shows up as an ``ERROR`` order.
- **Positions** are your coin balances (available plus reserved). The quote currency itself is cash, not a position. Bitkub does not report a cost basis, so ``avg_fill_price`` of a position is ``None``; use your own fills for P&L.
- **Portfolio value** is cash plus holdings valued at the last price in the quote currency. Coins with no market in that quote currency are left out of the valuation (and logged).

Open Orders
-----------

Bitkub lists open orders **per market**, so LumiBot polls the markets it knows about: markets you have traded, markets you hold coins in, and any listed in ``BITKUB_SYMBOLS``. An order placed by hand in the Bitkub app in a market you neither hold nor listed is not discovered until that market is in one of those sets.

Every order LumiBot tracks is also re-checked individually (``order-info``) on each poll, so fills, partial fills and cancellations are never inferred from absence in the open list. If any market cannot be read, the whole polling cycle is skipped rather than acting on a partial picture.

If a submission times out, LumiBot does **not** resend it. It looks for the order by its ``client_id`` in open orders and recent history; if it cannot find it, the order is reported as ``ERROR`` with an "order state unknown" message so you can check the Bitkub app.

Rate Limits
-----------

Bitkub blocks a client for 30 seconds (HTTP 429) after it exceeds an endpoint's limit. The client keeps its own request rate below the documented limits and refuses to send requests while a block is in force.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Bitkub error
     - Meaning and what to do
   * - 2, 3, 6
     - Missing/invalid API key or signature. Check ``BITKUB_API_KEY`` and ``BITKUB_API_SECRET``.
   * - 5
     - IP not allowed. Add this machine's IP to the key's allow-list.
   * - 8
     - Invalid timestamp. The client resyncs its clock from Bitkub and retries once automatically.
   * - 11
     - Invalid symbol. Symbols look like ``BTC_THB`` (base first).
   * - 15, "below minimum"
     - Order value too small. Increase the quantity.
   * - 18
     - Insufficient balance (remember THB reserved by open orders is not spendable).
   * - 52
     - The API key lacks the Trade permission.
   * - 61
     - The market is broker-sourced and cannot be traded through the API.
   * - 429
     - Rate limited; the client waits out the 30 second block.

Limitations
-----------

- No broker-side stop orders: if you need a stop-loss, the strategy must watch the price and sell, and it only protects you while it is running.
- REST polling (5 s default) rather than WebSocket streaming, so fill events can lag by up to one poll interval.
- Fill quantities for market buys are derived from Bitkub's per-fill history and should be confirmed with a small live order the first time you use the broker; positions and balances always come straight from your wallet.
