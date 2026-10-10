# LumiBot 4.6.9: what it can do (notes from reading its docs, 2026-10-09)

Sources: `repo\docsrc\*.rst`, `repo\llms.txt`, `repo\lumibot\components\agents\*`, https://lumibot.lumiwealth.com. Paths below are
relative to the LumiBot checkout (`<repo>`). "Not documented" = not found in those sources; verify before relying on it.

## Platform
- **Strategy lifecycle:** `initialize`, `before_market_opens`, `on_trading_iteration`, `before_market_closes`, `after_market_closes`,
  `on_filled_order`, `on_canceled_order`, `on_abrupt_closing`, `on_bot_crash`, `trace_stats`. `self.sleeptime = "1D"/"1H"/"15M"`.
  `self.vars` (state), `self.parameters`, `self.get_datetime()` (never `datetime.now()`), `self.log_message()`.
  `register_cron_callback("0 9 * * 1-5", fn)` exists but does nothing in backtests.
- **Orders:** market, limit, stop, stop-limit, trailing stop, bracket (`take_profit_price`+`stop_loss_price`), OTO, OCO. Helpers:
  `create_order`, `submit_order`, `cancel_open_orders`, `sell_all`, `close_position`. No `order_target*` helper.
  `DriftRebalancer` (`lumibot/example_strategies/drift_rebalancer.py`) rebalances to target weights, but it sells anything else in the account.
- **Backtest data:** Yahoo (free, daily only), Alpaca (stocks/ETFs/crypto, minute+day, your keys), Polygon, ThetaData, Databento (futures),
  CCXT (crypto), your own pandas/CSV. `BACKTESTING_DATA_SOURCE` env var OVERRIDES the class passed in code (`none` = use code).
  Fees via `buy_trading_fees=[TradingFee(percent_fee=...)]`. Outputs: tearsheet html/json, trades, indicators, stats, logs.
  `Strategy.backtest(DataClass, start, end, benchmark_asset="SPY", budget=50000)`.
- **Live:** brokers Alpaca, IBKR, Tradier, Schwab, Tradovate, ProjectX, Bitunix, Polymarket, CCXT crypto exchanges. Alpaca env vars:
  `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_IS_PAPER`. `run_live()` is a blocking, long-running process.
  BTC on Alpaca: `Asset("BTC","crypto")` with `quote=Asset("USD","forex")`, GTC only, needs `self.set_market("24/7")`.
  Whether stop/bracket orders work on Alpaca crypto: not documented (our own runner uses a stop-limit for BTC).
  Mixed stocks + BTC in one strategy: no documented pattern (design question: two strategies/processes?).
  Single-iteration mode for schedulers: `run_live(run_once=True)` / `LUMIBOT_SCHEDULED_EXECUTION` (built for BotSpot; not documented for local use).
  Restart: `self.vars` restored if `DB_CONNECTION_STR` set; positions/orders re-synced from the broker.
- **Not built in:** risk limits (no max position / drawdown halt / circuit breaker), a local live dashboard, Windows service supervision.
  BotSpot (Lumiwealth's hosted product) has hosting, alerts and a kill switch; pricing is not in the docs.
- **Notifications:** Discord (`DISCORD_WEBHOOK_URL`), Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`); Slack/email providers exist in code.
- **CLI:** `lumibot demo`, `lumibot init <p>`, `lumibot backtest <p> --days N`, `lumibot run <p> --paper`.

## AI agents (`lumibot/components/agents/`, docs `docsrc/agents*.rst`)
- `self.agents.create(name=, system_prompt=, model=, allow_trading=, allow_network=, tools=, mcp_servers=, rules_path=, ...)` in `initialize()`;
  `self.agents["x"].run(task_prompt=, context={...})` inside `on_trading_iteration()`. **`allow_trading` defaults to True**: set False for researchers.
  Multi-agent = plain Python (pass `result.summary` forward via `context=`); `run_together([...])` for parallel calls.
- **Credentials:** an LLM API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, ...) or the BotSpot-issued gateway
  token. **A Claude subscription/Claude Code login is NOT supported.** Default model `openai/gpt-6-luna`; Anthropic ids like
  `anthropic/claude-sonnet-4-6` work. Luna ~ $0.10/M input, $0.50/M output tokens (internal audit note). Caps: `LUMIBOT_AGENT_MAX_MODEL_CALLS`.
- **78 built-in tools:** account/orders, market data, indicators, options, FRED macro (`FRED_API_KEY`), SEC filings, Alpaca news, web/browser
  (`allow_network=True`), memory (persists in `.lumibot\memory\...\memory.sqlite`), notify (Telegram/email/Slack, opt-in), docs search, 4 skills.
  Custom tools: `@agent_tool` on a Strategy method, or MCP servers.
- **Backtesting agents:** look-ahead is blocked mechanically for LumiBot-owned data (prices, indicators, SEC, FRED, news), NOT for open web.
  The LLM itself may know post-window facts. Replay cache makes reruns free/repeatable; "a cache hit is not independent validation".
  Docs: "a successful run does not establish profitable trading." Example tearsheets are only 1-3 weeks long.
- **Guardrails:** order-readiness gate (must fetch portfolio/positions/open orders/last price before an order). No max position size, drawdown
  stop, or human approval step: put hard limits in deterministic Python around the agent (the docs' "risk gate" pattern).
- **Autonomy:** no agent scheduling beyond the strategy's `sleeptime`; memory/theses persist; observability via `*_agent_detail.parquet`, traces.
- **Simplest credible demos:** `example_strategies/agent_macro_risk.py` (trend TQQQ/SHV), `agent_news_sentiment.py`, `agent_momentum_allocator.py`,
  `ai_researcher_trader.py` (read-only researcher + trader that owns risk).
