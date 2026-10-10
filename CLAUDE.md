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
