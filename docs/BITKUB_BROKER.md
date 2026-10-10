# BITKUB_BROKER.md

> Engineering notes for the Bitkub spot-crypto broker and data source: verified API behavior, design decisions, and what still needs a live check.

**Last Updated:** 2026-10-10
**Status:** Active. Public market-data endpoints verified live; private (key-required) endpoints implemented from the official docs and covered by deterministic tests, **not yet validated against a live account**.
**Audience:** Developers, AI Agents

---

## Overview

Bitkub is a Thai spot exchange (THB and USDT quote markets). Code map:

| Piece | File |
|---|---|
| REST client (signing, errors, rate limits, formatting) | `lumibot/tools/bitkub_helpers.py` (`BitkubClient`) |
| Market data (`get_last_price`, `get_quote`, `get_historical_prices`, order book, symbols) | `lumibot/data_sources/bitkub_data.py` (`BitkubData`) |
| Broker (orders, cancel, balances/positions, polling stream) | `lumibot/brokers/bitkub.py` (`Bitkub`) |
| Credentials / selection | `lumibot/credentials.py` (`BITKUB_CONFIG`, `BROKER=bitkub`, `DATA_SOURCE=bitkub`) |
| Strategy default quote asset (THB/USDT instead of USD) | `lumibot/strategies/_strategy.py` |
| Backtesting data source (completed-candle visibility, current-candle fills, cached chunked download) | `lumibot/data_sources/bitkub_backtesting_data.py`, `lumibot/backtesting/bitkub_backtesting.py`, `lumibot/tools/bitkub_history.py` |
| Tests + recorded fixtures | `tests/test_bitkub_helpers.py`, `tests/test_data_source_bitkub.py`, `tests/test_broker_bitkub.py`, `tests/test_bitkub_strategy_defaults.py`, `tests/fixtures/bitkub/` |
| User guide | `docsrc/brokers.bitkub.rst` |

Official API reference (authoritative): the `bitkub/bitkub-official-api-docs` repository (`rest-v3.md`, `rest-v4.md`, `rest-v1.md`, websocket docs) and the developer center at `api.bitkub.com/docs` (a client-rendered page that scrapers see as an empty shell; read the markdown in the repository instead).

## Verified live (public endpoints, 2026-10-10)

Probed with unauthenticated requests; recorded responses are the fixtures under `tests/fixtures/bitkub/`.

| Finding | Docs say | Live behavior | How the code handles it |
|---|---|---|---|
| Symbol format | `btc_thb` in tables, `thb_btc` in the place-bid example | `BTC_THB`, **case-insensitive**; `thb_btc` returns `{"error": 11}` | `format_symbol`/`parse_symbol`; broker sends lowercase `base_quote` |
| Errors can arrive with HTTP 200 | n/a | `{"error": 11}` with HTTP 200; private endpoints answer HTTP 400 `{"error": 2, "result": null}` when unauthenticated | `_unwrap` always parses the body; never trusts status alone |
| Two response envelopes | v3 `{error,result}`; v4 `{code,message,data}` | Also true for `/tradingview/history` errors: HTTP 400 `{"code":"7002","message":"Symbol not found"}` | `_unwrap` handles both |
| Bare payloads | n/a | `servertime` is a bare integer (ms); `ticker` and `status` are bare lists | passed through |
| Trade timestamps | milliseconds in the example | **seconds** in `/api/v3/market/trades` | not used for pricing; documented |
| Candle history | `/tradingview/history`, resolutions `1,5,15,60,240,1D` | Timestamps are seconds, UTC-aligned; large windows work (10 years of daily = 3,071 bars; 30 days of 1-minute = 37k bars); the **still-forming candle is included** | completed bars only (`t + interval <= now`) |
| Sparse candles | n/a | Minutes without trades are absent (gaps up to hours on illiquid markets) | never filled in; window widened 1x/4x/16x/64x to find enough real candles |
| **Conflicting duplicate candles** | n/a | The same timestamp appears twice with *different* volume/high/close, even for a one-candle query (stable across repeated calls, row order varies). Example 2025-04-25 daily: `v=48.9, close 3,137,364` vs `v=101.6, close 3,195,139` | keep the highest-volume row. Evidence: the larger row has a superset price range and its close matches the next candle's open (3,195,139 vs 3,195,842 next open; the other row is 1.9% off). Pinned by `test_conflicting_duplicate_candles_keep_the_most_complete_version` in both row orders |
| Markets | n/a | 474 symbols: 366 active; `source` is `exchange` (337) or `broker` (137); quote assets THB (464) and USDT (10) | trade only `status == active` and `source == exchange` (the trading endpoints return error 61 for broker coins) |
| Precision fields | `price_step`, `quantity_step`, ... | `quantity_step`/`quantity_scale` look like a quote-side unit (BTC shows `1` / `0`), so they are **not** used for coin sizing | coins use `base_asset_scale`, THB uses `quote_asset_scale`, prices use `price_scale`, minimum is `min_quote_size` |

## Implemented from the docs, NOT yet verified against a live account

These need a keyed validation by the account owner (see the checklist below). The code is deliberately conservative about all of them.

1. **Signature acceptance.** The algorithm is `hex(HMAC_SHA256(secret, timestamp + METHOD + path[?query] + body))` and is pinned against independent `openssl` vectors, but only Bitkub's servers can confirm they accept it (especially the "query string is part of the signed path" reading for GETs).
2. **`amt` semantics.** Buy `amt` = quote currency (THB) to spend; sell `amt` = coin quantity. Numbers must have no trailing zeros, so request bodies are built by hand (`build_json_body`).
3. **Fill accounting from `order-info.history`.** Buy fills are in THB: `coins = (amount - fee) / rate` (consistent with the docs' open-order example: amount 100, fee 0.25, rate 10, receive 9.975). Sell fills are in coins. Positions and cash always come from wallet balances, never from this arithmetic, so an error here can only mis-state a fill price/quantity, not holdings.
4. **Order status mapping.** `order-info.status` is `filled | unfilled | cancelled`, with `partial_filled` for partial fills. The open-orders list carries no status or fill data.
5. **Private symbol case.** Private endpoints are sent lowercase (`btc_thb`) per the docs' parameter tables; the failure mode if wrong is a harmless error 11.
6. **`client_id` constraints** (length/charset) are undocumented; the broker uses `lmbot-<ms>-<6 hex>`.

## Design decisions

- **No stop orders.** Bitkub has none. They are rejected loudly (never approximated) so a strategy cannot believe it is protected.
- **Market buy sizing.** `Order` has no notional field, so a market buy spends `quantity x best ask` (fresh ticker). The tracked quantity is the executable size, and is replaced by the real filled quantity when the order completes.
- **Fills are increments.** `Order.add_transaction` is additive, so polling computes `cumulative - already booked` and dispatches only the delta (the reference Bitunix polling passes the order total, which would double count after a partial fill).
- **Absence from the open list never means cancelled.** A filled market order and a lagging list both look like "missing". Every tracked active order is settled through `order-info` each poll.
- **Incomplete reads abort the cycle.** Open orders are per symbol; if any symbol cannot be read, the poll is skipped. Balances that cannot be read raise instead of returning an empty (flat) account, because the shared position sync prunes absent assets.
- **No blind retries on POST.** After a timeout the order may exist. The broker reconciles by `client_id` (open orders, then recent history) and otherwise reports an explicit "state unknown" error. GETs retry with backoff; a bad-timestamp rejection (error 8) resyncs the clock and retries once, since the exchange rejects it before touching the book.
- **A failed cancel does not mark the order errored.** It checks the real state instead (it may already be filled).
- **Cash is the total quote balance** (CCXT precedent), so THB reserved by open buys is counted; the exchange rejects an unaffordable buy (error 18).
- **Strategy default quote asset** follows the broker (THB/USDT) like Bitunix's USDT special case; USD markets do not exist on Bitkub.

## Backtesting

`BitkubBacktesting` downloads candles lazily (chunked, cached; coverage is recorded only after every request succeeded) and serves
them with the CCXT crypto model: strategies see completed candles only; fills use the candle that starts now. Why not `PandasData`:
it exposes the candle labelled "now" with its full close, so UTC-start-stamped daily candles leak a day of future into signals
(reproduced on synthetic data). The engine's CCXT-keyed branches were extended to `BITKUB` (4 in `backtesting_broker.py`, 1 in
`_strategy.py`). Candles have no bid/ask, so fills carry no spread unless the caller adds slippage; fees (0.25%/side on a standard
account) must be passed to `backtest()`.

## Known gaps / future work

- WebSocket streaming (`/api/v3/market/wstoken`, order/match updates, public ticker/order book) for lower-latency fills.
- Emulated stop orders (client-side) if a user needs them, clearly labelled as not broker-protected.
- Discovering open orders in markets the account neither holds nor has traded without `BITKUB_SYMBOLS`.

## First live validation checklist (account owner)

Run in this order and stop at the first surprise. Use a key with **only** the permissions needed for each step and never Deposit/Withdraw.

1. Read-only key: `BitkubClient(...).get_balances()` and `BitkubData().get_symbols()`; confirm no signature error (this validates item 1 above).
2. Construct `Bitkub(BITKUB_CONFIG, connect_stream=False)` and call `_pull_positions(None)` and `_get_balances_at_broker(THB, None)`; compare with the Bitkub app.
3. With a Trade-enabled key, place a **limit buy far below the market** for roughly the minimum order (about 10 THB) so it cannot fill; confirm it appears in `get_open_orders`, then cancel it with `cancel_order` and confirm `order-info` says cancelled.
4. Only then, a minimum-size **market** buy and sell of a liquid coin: confirm the filled quantity/price LumiBot books matches the Bitkub app (this validates item 3), and that the wallet balance matches `_pull_positions`.
