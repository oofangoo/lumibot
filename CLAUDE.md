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
- Keep `..\repo` pinned at LumiBot v4.6.9 (commit 50894f8d, see `requirements.lock.txt`) for the whole paper test. We have not modified LumiBot
  itself (the bracket-order workaround lives in `strategies\lumi_alpaca_compat.py`); upgrade only deliberately, then re-validate with a parity backtest.
- The journals stay local while the remote is public. If the user wants them versioned, they should create a private repo for them.
