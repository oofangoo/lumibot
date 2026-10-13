# Workspace instructions for Claude Code

This folder runs the user's Lumibot paper-trading experiments (Alpaca PAPER account, $50k budget).

**Before doing anything, read `DIARY.md` (context, rules, how to resume) and `EXPERIMENT_LOG.md` (results).**

Hard rules:
- Never place real-money orders or change `ALPACA_IS_PAPER=true`. Moving to real money is the user's decision, after 4-8 weeks of paper results.
- Never read or print the values in `.env`. Use `research\check_alpaca.py` to verify connectivity.
- Only SPY, QQQ, SHV and BTC/USD belong to the bot; never touch other positions in the account.
- Run scripts via `.\run.ps1 <script>` from this folder (sets the Avast CA bundle and the venv).
- Log each experiment in `EXPERIMENT_LOG.md` and each action in the "Action log" of `DIARY.md`.
- Be honest about results: say so when a backtest fails or an expectation is lowered.

## Leverage LumiBot first (standing rule from the user, 2026-10-10: "make sure we are leveraging LumiBot as much as possible ... always")
This workspace exists to exercise LumiBot. Use its own facilities before writing anything by hand, and say in the reply which ones were used.
- **Charts**: `Strategy.add_line` / `add_marker` / `add_ohlc` (use `dt=` to place points at other dates, `plot_name=` for subplots), run through a
  LumiBot backtest (`BitkubBacktesting` for Bitkub data) so `plot_indicators` writes the Plotly `*_indicators.html`. Also `plot_returns` (trades chart) and the
  QuantStats tearsheet. Do not hand-draw SVG/HTML charts unless LumiBot cannot express the picture; if so, state what it lacks.
- **Data and backtests**: LumiBot data sources and `run_backtest`; `self.indicators` for SMA/RSI/etc.; broker classes for live. Plain pandas only where LumiBot
  has no equivalent (multi-coin cross-sectional studies, percentile statistics) or as a deliberate independent cross-check, and say which.
- **Missing capability**: prefer adding it to the LumiBot fork (`..\lumibot-dev`, topic branch, tests, docs) over working around it.
- Before building anything, search `..\lumibot-dev\lumibot` and `docsrc` for an existing facility.

## Code management (since 2026-10-10)
This folder is its own git repo on branch `trading-workspace`. `origin` is the user's **public** fork (https://github.com/oofangoo/lumibot.git);
the branch has unrelated history from the fork's `dev` (which mirrors LumiBot). `..\repo` is the LumiBot clone: `origin` = the fork, `upstream` = Lumiwealth.
- **The live bot runs this working tree** (Task Scheduler executes it directly). Keep it clean and on `trading-workspace`. Do changes on a
  topic branch (`fix/...`, `feat/...`, `research/...`), merge with `--no-ff` once verified, and tag each deployment `live-YYYY-MM-DD[-n]`.
  Do not leave the tree on a topic branch or with uncommitted edits to `strategies/` across a scheduled run (stocks 22:00 weekdays, BTC 22:05
  daily, Bangkok time). The dashboard shows the deployed commit and warns about uncommitted live code; `logs\run_status.csv` records the commit per run.
- Commits: `type(scope): why` (feat / fix / chore / docs / research), body explains cause and how it was verified; end with the co-author trailer.
  A change to trading logic must say how it was verified (backtest id, live no-trade probe, ...). Reference experiment ids (E0xx).
- **Never commit** `.env`, `DIARY.md`, `EXPERIMENT_LOG.md`, `state/`, `logs/`, `data/`, absolute personal paths or credentials: the remote is public.
  `.githooks/pre-commit` blocks them (enable on a fresh clone with `git config core.hooksPath .githooks`). Never use `--no-verify`.
- Never force-push or rewrite pushed history. Push with `git push origin trading-workspace --follow-tags`.
- Keep `..\repo` pinned at LumiBot v4.6.18 (commit 730d5608, upgraded from v4.6.9 on 2026-10-13 after an identical-result parity backtest; see `requirements.lock.txt`) for the whole paper test. We have not modified LumiBot
  itself (the bracket-order workaround lives in `strategies\lumi_alpaca_compat.py`); upgrade only deliberately, then re-validate with a parity backtest.
- The journals stay local while the remote is public. If the user wants them versioned, they should create a private repo for them.
