"""Download 5-minute bars (split-adjusted) for SPY and QQQ from Alpaca into data/*.parquet. Read-only market data.
Tries the consolidated SIP feed first (free plan allows it when >15 min old), falls back to IEX."""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402

c = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
(ROOT / "data").mkdir(exist_ok=True)
end = datetime.now(timezone.utc) - timedelta(minutes=20)
tf = TimeFrame(5, TimeFrameUnit.Minute)


def fetch(sym, feed, start, stop):
    r = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start, end=stop,
                                          adjustment=Adjustment.ALL, feed=feed))
    return r.df


for sym in ("SPY", "QQQ"):
    feed = DataFeed.SIP
    try:
        fetch(sym, feed, end - timedelta(days=5), end)
    except Exception as e:
        print(f"{sym}: SIP not available ({type(e).__name__}); using IEX")
        feed = DataFeed.IEX
    parts = []
    for yr in range(2016, end.year + 1):
        a = datetime(yr, 1, 1, tzinfo=timezone.utc)
        b = min(datetime(yr + 1, 1, 1, tzinfo=timezone.utc), end)
        try:
            d = fetch(sym, feed, a, b)
        except Exception as e:
            print(f"{sym} {yr}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if len(d):
            parts.append(d.reset_index(level=0, drop=True))
        print(f"{sym} {yr}: {len(d)} bars", file=sys.stderr)
    df = pd.concat(parts).sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df = df.between_time("09:30", "15:59")
    df.to_parquet(ROOT / "data" / f"{sym}_5min_{feed.value}.parquet")
    print(f"{sym}: feed={feed.value} rows={len(df)} range={df.index[0]} .. {df.index[-1]}")
