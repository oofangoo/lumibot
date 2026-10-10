"""E019 page: LumiBot's own indicators chart (btc_cycle_chart.py) wrapped in a short explanation.

Reads the newest logs/BtcCycleChart_*_indicators.html (written by LumiBot's plot_indicators) and logs/BtcCycleChart_latest_summary.json, keeps the
engine's Plotly figure untouched (apart from its title text) and writes one self-contained HTML page. Every number in the text comes from the summary,
so the words and the chart cannot disagree.
Run:  .\\run.ps1 research\\btc_cycle_page.py <out.html>
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"
ENGINE_TITLE = "Indicators for BtcCycleChart Strategy Indicators"
NEW_TITLE = "BTC/THB: price, falls from the high, and the cycle replay"
MINUS = "−"


def thb(x):
    return f"{x:,.0f}"


def pct(x, signed=True, digits=0):
    text = f"{abs(x) * 100:.{digits}f}%"
    return (MINUS if x < 0 else ("+" if signed else "")) + text


def day(iso):
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def months(days):
    return days / 30.44


def engine_body(path):
    html = path.read_text(encoding="utf-8")
    assert html.count(ENGINE_TITLE) == 1, "engine title text not found exactly once"
    html = html.replace(ENGINE_TITLE, NEW_TITLE)
    inner = html[html.index("<body>") + len("<body>"): html.rindex("</body>")]
    # plotly.js contains the literal U+FFFD in the regex /<U+FFFD>/g (it strips replacement characters). Spell it as the JS escape, which means the
    # same thing in a regex literal, so the published file has no raw replacement character.
    raw, escaped = "/" + chr(0xFFFD) + "/g", "/" + chr(92) + "ufffd/g"
    assert inner.count(raw) == 1, "expected exactly one literal U+FFFD regex in plotly.js"
    inner = inner.replace(raw, escaped)
    assert "<script" in inner and "plotly-graph-div" in inner
    assert not re.search(r'<script[^>]+src="https?:', inner), "engine page loads an external script"
    return inner


def cycle_rows(s, last_day):
    rows = []
    for k, c in enumerate(s["cycles"]):
        current = k == len(s["cycles"]) - 1
        back = (f"{day(c['back_day'])}<span class='sub'>after {c['days_to_back']} days ({c['days_to_back'] / 365.25:.1f} years)</span>" if c["back_day"]
                else f"Not yet<span class='sub'>day {(date.fromisoformat(last_day) - date.fromisoformat(c['top_day'])).days} since the top</span>")
        row_class = " class='now'" if current else ""
        rows.append(
            f"<tr{row_class}><th scope='row'>{day(c['top_day'])}<span class='sub'>{thb(c['top_price'])} THB</span></th>"
            f"<td>{day(c['low_day'])}{' (so far)' if current else ''}<span class='sub'>{thb(c['low_price'])} THB</span></td>"
            f"<td class='num neg'>{pct(c['depth'])}</td><td class='num'>{c['days_to_low']}<span class='sub'>{months(c['days_to_low']):.0f} months</span></td><td>{back}</td></tr>")
    return "\n".join(rows)


def level_rows(s):
    last, top = s["last_close"], s["top_price"]
    now, past = s["cycles"][-1], s["cycles"][:-1]
    items = [
        ("All-time high", top, "Above this, the new-high group is in play."),
        ("200-day average", s["sma200"], "The long-term trend line. A trend rule that follows it moves to cash below it."),
        ("50-day average", s["sma50"], "The short-term trend line. Price crossed it often in the past, so a close on either side says little."),
        (f"Low of this cycle so far ({day(now['low_day'])})", now["low_price"], "Closing below it makes a new low for this cycle."),
    ]
    for c in past:
        items.append((f"A repeat of the {date.fromisoformat(c['top_day']).year} fall ({pct(c['depth'])} from the top)", top * (1 + c["depth"]),
                      f"Where price would be if this cycle fell as far as the cycle that topped in {date.fromisoformat(c['top_day']).year}."))
    items.sort(key=lambda it: -it[1])
    out = []
    for name, level, why in items:
        out.append(f"<tr><th scope='row'>{name}</th><td class='num'>{thb(level)}</td><td class='num {'pos' if level > last else 'neg'}'>{pct(level / last - 1)}</td><td>{why}</td></tr>")
    return "\n".join(out)


def build(s, body):
    last, top = s["last_close"], s["top_price"]
    now, past = s["cycles"][-1], s["cycles"][:-1]
    day_n = (date.fromisoformat(s["today"]) - date.fromisoformat(now["top_day"])).days
    f365, c365 = s["fan"]["365"], s["fan_pct_change"]["365"]
    o_new, o_all = s["outcomes_since_mid_2021"], s["outcomes_all_history"]
    bh = s["buy_and_hold"]
    p2019, p2021 = past[0], past[1]
    fall19, fall21 = top * (1 + p2019["depth"]), top * (1 + p2021["depth"])
    subs = {
        "TODAY": day(s["today"]), "LAST": thb(last), "DAYN": str(day_n), "FROMHIGH": pct(last / top - 1), "FROMLOW": pct(last / now["low_price"] - 1), "FROMLOW_U": pct(last / now["low_price"] - 1, signed=False),
        "LOWDAY": day(now["low_day"]), "LOWPX": thb(now["low_price"]), "TOPDAY": day(now["top_day"]), "TOPPX": thb(top),
        "VS50": pct(last / s["sma50"] - 1, digits=1), "VS200": pct(last / s["sma200"] - 1, digits=1), "VS200_U": pct(last / s["sma200"] - 1, signed=False, digits=1), "S50": thb(s["sma50"]), "S200": thb(s["sma200"]),
        "CYCLES": cycle_rows(s, s["today"]), "LEVELS": level_rows(s), "BODY": body,
        "M1": f"{months(p2019['days_to_low']):.0f}", "M2": f"{months(p2021['days_to_low']):.0f}",
        "Y1": f"{p2019['days_to_back'] / 365.25:.1f}", "Y2": f"{p2021['days_to_back'] / 365.25:.1f}",
        "D19": str(p2019["days_to_low"]), "D21": str(p2021["days_to_low"]), "DNOW": str(now["days_to_low"]),
        "WEAK": thb(f365[0]), "WEAKP": pct(c365[0]), "TYP": thb(f365[1]), "TYPP": pct(c365[1]), "STRONG": thb(f365[2]), "STRONGP": pct(c365[2]),
        "O1": f"{o_new['down_20'] * 100:.0f}%", "O2": f"{o_new['middle'] * 100:.0f}%", "O3": f"{o_new['new_high'] * 100:.0f}%",
        "A1": f"{o_all['down_20'] * 100:.0f}%", "A2": f"{o_all['middle'] * 100:.0f}%", "A3": f"{o_all['new_high'] * 100:.0f}%",
        "NYRS": f"{o_new['about_independent_years']:.1f}", "NYRS_ALL": f"{o_all['about_independent_years']:.1f}", "MED_ALL": pct(s["all_history_median_365"]),
        "FALL19": thb(fall19), "FALL21": thb(fall21), "FALL19P": pct(fall19 / last - 1), "FALL21P": pct(fall21 / last - 1),
        "FALL19D": pct(p2019["depth"]), "FALL21D": pct(p2021["depth"]), "NEWHIGHP": pct(s["new_high_gain"]),
        "BH_TOTAL": pct(bh["total_return"]).replace("+", "+"), "BH_CAGR": f"{bh['cagr'] * 100:.0f}%", "BH_DD": pct(-bh["max_drawdown"]), "BH_VOL": f"{bh['volatility'] * 100:.0f}%",
        "FALL_RANGE": f"{abs(max(c['depth'] for c in s['cycles'])) * 100:.0f}% to {abs(min(c['depth'] for c in s['cycles'])) * 100:.0f}%",
        "TREND_PRICE_DAYS": str(s["n_days"]),
    }
    page = TEMPLATE
    for key, value in subs.items():
        page = page.replace(f"@@{key}@@", value)
    leftover = re.findall(r"@@[A-Z0-9_]+@@", page)
    assert not leftover, f"unfilled placeholders: {leftover}"
    return page


TEMPLATE = r"""<title>BTC Cycle Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column; facts strip, pattern table, LumiBot's chart at full width, three scenarios side by side, levels table, notes. Dark to match LumiBot's chart. */
:root {
  --bg: #0e1013; --panel: #15181d; --line: #262b33; --fg: #e8ecf0; --muted: #98a2ae; --faint: #6c7682;
  --gold: #ffd43b; --red: #ff6b6b; --green: #69db7c; --grey: #ced4da; --blue: #74c0fc; --violet: #b197fc;
  --display: "Bricolage Grotesque", "IBM Plex Sans", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 400 16px/1.6 var(--body); padding-inline: clamp(16px, 4vw, 40px); padding-block: 40px 64px; }
.wrap { max-width: 1180px; margin-inline: auto; display: flex; flex-direction: column; gap: 56px; }
h1, h2, h3 { font-family: var(--display); font-weight: 700; line-height: 1.1; margin: 0; text-wrap: balance; }
h1 { font-size: clamp(40px, 7vw, 68px); letter-spacing: -0.02em; }
h2 { font-size: clamp(24px, 3.4vw, 32px); letter-spacing: -0.01em; }
h3 { font-size: 20px; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 500 12px/1.4 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.lede { font-size: clamp(17px, 2.2vw, 20px); color: var(--fg); max-width: 62ch; }
header { display: flex; flex-direction: column; gap: 18px; }
section { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
.muted { color: var(--muted); }
.sub { display: block; font: 400 12px/1.4 var(--mono); color: var(--muted); margin-top: 2px; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pos { color: var(--green); } .neg { color: var(--red); }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0; border-block: 1px solid var(--line); }
.fact { padding: 16px 20px 16px 0; display: flex; flex-direction: column; gap: 4px; }
.fact .label { font: 500 12px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.fact .value { font: 500 26px/1.2 var(--mono); font-variant-numeric: tabular-nums; }
.fact .note { font-size: 13px; color: var(--muted); }

.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 15px; }
th, td { text-align: left; padding: 12px 16px 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { font: 500 12px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); border-bottom-color: var(--faint); }
tbody th { font-weight: 500; }
tr.now th, tr.now td { color: var(--gold); }
tr.now .sub { color: var(--muted); }
tr.now td.neg { color: var(--gold); }

.plotscroll { overflow-x: auto; max-width: 100%; border: 1px solid var(--line); background: #111111; }
.plotframe { min-width: 980px; }
.guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; }
.guide div { display: flex; flex-direction: column; gap: 6px; font-size: 15px; color: var(--muted); }
.guide b { font: 500 13px/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--fg); }

.scen { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 32px; }
.sc { display: flex; flex-direction: column; gap: 12px; padding-top: 16px; border-top: 1px solid var(--line); min-width: 0; }
.swatch { display: inline-block; width: 34px; height: 4px; margin-right: 10px; vertical-align: middle; }
.sc .big { font: 500 30px/1.15 var(--mono); font-variant-numeric: tabular-nums; }
.sc .big span { font-size: 15px; color: var(--muted); margin-left: 6px; }
.share { display: flex; gap: 20px; flex-wrap: wrap; font: 400 13px/1.4 var(--mono); color: var(--muted); }
.share b { color: var(--fg); font-weight: 500; font-size: 18px; display: block; }
.read { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 28px 40px; }
.read p { color: var(--fg); }
.fine { font-size: 13px; color: var(--muted); max-width: 82ch; }
.fine a { color: var(--muted); }
footer { display: flex; flex-direction: column; gap: 10px; border-top: 1px solid var(--line); padding-top: 24px; }
@media (max-width: 560px) { .fact { padding-right: 12px; } .fact .value { font-size: 22px; } }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Bitkub BTC/THB &middot; daily candles to @@TODAY@@ &middot; drawn with LumiBot</div>
  <h1>BTC Cycle Map</h1>
  <p class="lede">Each of the last three Bitcoin tops (2019, 2021, 2025) was followed by a fall of @@FALL_RANGE@@ (the latest is the shallowest so far). The latest top was @@TOPDAY@@ and today is day @@DAYN@@ since it. This page shows that pattern, where BTC stands now, and three ways the next year could go, using only what past years actually did.</p>
</header>

<div class="facts">
  <div class="fact"><span class="label">BTC today</span><span class="value">@@LAST@@</span><span class="note">THB, close of @@TODAY@@</span></div>
  <div class="fact"><span class="label">From the high</span><span class="value neg">@@FROMHIGH@@</span><span class="note">high @@TOPPX@@ on @@TOPDAY@@</span></div>
  <div class="fact"><span class="label">From this cycle's low</span><span class="value pos">@@FROMLOW@@</span><span class="note">low @@LOWPX@@ on @@LOWDAY@@</span></div>
  <div class="fact"><span class="label">Trend</span><span class="value">Above both</span><span class="note">50-day average @@VS50@@, 200-day @@VS200@@</span></div>
</div>

<section>
  <h2>The pattern</h2>
  <p>Bitcoin moves in long boom-and-bust cycles. A cycle top here is an all-time high that stood for a year or more. Bitkub's candle history holds three.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Cycle top</th><th>Lowest point after it</th><th>Fall</th><th>Days to the low</th><th>Back at the old top</th></tr></thead>
    <tbody>
@@CYCLES@@
    </tbody>
  </table>
  </div>
  <p>In the first two cycles the low came about @@M1@@ and @@M2@@ months after the top, and the old top took @@Y1@@ and @@Y2@@ years to win back. This cycle's low so far came on the same schedule as the first (@@DNOW@@ days against @@D19@@) but was shallower, and price has climbed @@FROMLOW_U@@ since.</p>
  <p class="fine">The 2017 top is older than Bitkub's candle history. In dollars it fell 83% in the year after the top (BTC-USD, Yahoo, from my earlier analysis E018).</p>
</section>

<section>
  <h2>The chart</h2>
  <div class="guide">
    <div><b>1 &middot; Price (log scale)</b>Equal percentage moves look equal. Dotted blue is the 50-day average and dashed orange the 200-day. Red triangles mark cycle tops, green triangles cycle lows. The red, grey and green lines right of today are what past years did, applied to today's price. They are not a forecast.</div>
    <div><b>2 &middot; Fall from the high</b>How far below its all-time high BTC stood on each day. Every cycle shows the same shape: a drop, a low, a climb back to zero.</div>
    <div><b>3 &middot; Cycle replay</b>The 2019 cycle (blue) and the 2021 cycle (violet) shifted onto the @@TOPDAY@@ top, with this cycle in yellow. Dates on the axis belong to this cycle. Hover any point to see its real date.</div>
  </div>
  <div class="plotscroll"><div class="plotframe">
@@BODY@@
  </div></div>
  <p class="fine">Drag across any panel to zoom, use the buttons above each panel for 1 month to all, and click a legend entry to hide or show that line.</p>
</section>

<section>
  <h2>Three ways the next year could go</h2>
  <p>Take every day since mid-2021 as a starting point and look at where BTC stood a year later. Those years fall into three groups. The line for each group is drawn on panel 1 above.</p>
  <div class="scen">
    <div class="sc">
      <h3><span class="swatch" style="background:var(--red)"></span>Another leg down</h3>
      <p class="muted">Down more than 20% after a year.</p>
      <div class="big">@@WEAK@@<span>THB &middot; @@WEAKP@@</span></div>
      <div class="share"><div><b>@@O1@@</b>of years since mid-2021</div><div><b>@@A1@@</b>of all years</div></div>
      <p>The red line is the 10th-percentile year. It sits under this cycle's low (@@LOWPX@@). A repeat of the 2019 fall would put price at @@FALL19@@ (@@FALL19P@@ from today) and a repeat of the 2021 fall at @@FALL21@@ (@@FALL21P@@).</p>
    </div>
    <div class="sc">
      <h3><span class="swatch" style="background:var(--grey)"></span>Grind</h3>
      <p class="muted">Between &minus;20% and back at the old high.</p>
      <div class="big">@@TYP@@<span>THB &middot; @@TYPP@@</span></div>
      <div class="share"><div><b>@@O2@@</b>of years since mid-2021</div><div><b>@@A2@@</b>of all years</div></div>
      <p>The grey line is the median year since mid-2021. Across all of Bitkub's history the median year was @@MED_ALL@@ (dotted grey). Either way price ends the year under the old high of @@TOPPX@@.</p>
    </div>
    <div class="sc">
      <h3><span class="swatch" style="background:var(--green)"></span>New high</h3>
      <p class="muted">Above the old high, a gain of @@NEWHIGHP@@ or more.</p>
      <div class="big">@@STRONG@@<span>THB &middot; @@STRONGP@@</span></div>
      <div class="share"><div><b>@@O3@@</b>of years since mid-2021</div><div><b>@@A3@@</b>of all years</div></div>
      <p>The green line is the 90th-percentile year. Those years came with the 2020-21 and 2023-24 runs.</p>
    </div>
  </div>
  <p class="fine">These are shares of past start days, not probabilities. Since mid-2021 the sample holds about @@NYRS@@ independent years (@@NYRS_ALL@@ across all history), because neighbouring start days move together. The outer two groups are both common, so a typical year ended far from where it began.</p>
</section>

<section>
  <h2>Levels to watch</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>Level</th><th>THB</th><th>From today</th><th>Why it matters</th></tr></thead>
    <tbody>
@@LEVELS@@
    </tbody>
  </table>
  </div>
</section>

<section>
  <h2>How I read it</h2>
  <div class="read">
    <p>The pattern is real but thin. It rests on three cycles, and this one already differs from the first two: a shallower fall so far and a @@FROMLOW_U@@ rebound. The data cannot yet say whether the low is in. A repeat of the 2019 or 2021 fall would take price @@FALL19P@@ or @@FALL21P@@ from here.</p>
    <p>Nothing on this page is a buy or sell signal by itself. The 200-day average is the line a trend rule would follow, and price is @@VS200_U@@ above it today.</p>
    <p>Buying and holding from June 2018 returned @@BH_TOTAL@@ in LumiBot's backtest (@@BH_CAGR@@ a year, fee 0.25%), with a worst fall of @@BH_DD@@ and yearly volatility of @@BH_VOL@@. A buyer at the 2019 or 2021 top waited @@Y1@@ to @@Y2@@ years to get back to even.</p>
    <p>If you buy, size it for a drop of 50% to 75%, use money you will not need for about four years, and think about spreading the purchase over a few months. In past years spreading did not raise returns. It only softened the worst case.</p>
  </div>
</section>

<footer>
  <p class="fine"><b>How it was made.</b> A LumiBot strategy ran over real Bitkub BTC/THB daily candles (@@TREND_PRICE_DAYS@@ days) with the BitkubBacktesting data source. It reads the 50- and 200-day averages from <code>self.indicators</code>, draws with <code>add_line</code> and <code>add_marker</code>, and LumiBot's <code>plot_indicators</code> wrote the chart you see. The buy-and-hold figures are LumiBot's own backtest statistics. Two small additions were made to the LumiBot fork for this chart: <code>add_line</code> style and width are now drawn, and a subplot named "(log)" gets a log axis.</p>
  <p class="fine">Candle days are UTC. Cycle tops are all-time highs that stood for a year or more. Future lines apply the change seen over every past 365-day window to today's price. Replay lines are shifted in time to share one top date. Past results do not predict future results, and a Bitcoin position can lose most of its value. Research notes: E018 and E019 in the workspace experiment log.</p>
</footer>
</div>
"""


def main():
    out = Path(sys.argv[1])
    engine = max(LOGS.glob("BtcCycleChart_*_indicators.html"), key=lambda p: p.stat().st_mtime)
    summary = json.loads((LOGS / "BtcCycleChart_latest_summary.json").read_text(encoding="utf-8"))
    page = build(summary, engine_body(engine))
    out.write_text(page, encoding="utf-8")
    print(f"engine chart: {engine.name}\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
