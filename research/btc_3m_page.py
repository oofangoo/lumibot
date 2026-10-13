"""E021 page: LumiBot's indicators chart (btc_3m_chart.py) wrapped in a plain-language explanation of the 91-day forecast.

Reads logs/btc_3m_forecast.json (written by btc_3m_scenarios.py) and the newest logs/Btc3mForecast_*_indicators.html (written by LumiBot's plot_indicators).
Every number in the text comes from the JSON, so the words and the chart cannot disagree.
Run:  .\\run.ps1 research\\btc_3m_page.py <out.html>
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"
ENGINE_TITLE = "Indicators for Btc3mForecast Strategy Indicators"
NEW_TITLE = "BTC/THB: price and the next 91 days"
MINUS = "&minus;"
NAMES = {
    "EMP": "Plain history", "RW": "Random walk", "BOOT0": "Bootstrap, no drift", "BOOT": "Bootstrap, past drift", "FHS": "Volatility simulation",
    "QLIN": "Machine learning, linear", "QGBM": "Machine learning, boosted trees", "ENS": "Average of all seven",
}
WHAT = {
    "EMP": "Every past 91-day window in Bitkub's history, as it happened.",
    "RW": "A bell curve around zero, sized by how calm or wild the last weeks were.",
    "BOOT0": "Shuffles real daily moves in 10-day blocks, with the long-run upward drift removed.",
    "BOOT": "The same shuffle, keeping the upward drift of the past 8 years.",
    "FHS": "Shuffles real moves, then lets volatility rise and fall along each path.",
    "QLIN": "Learns from trend, momentum, drawdown and volatility how wide and where the range sits.",
    "QGBM": "The same inputs, with decision trees instead of a straight line.",
    "ENS": "Takes the middle of all seven forecasts.",
}


def pct(x, signed=True, digits=0):
    text = f"{abs(x) * 100:.{digits}f}%"
    return (MINUS if round(x * 100, digits) < 0 else ("+" if signed else "")) + text


def thb(x):
    return f"{x:,.0f}"


def day(iso):
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def engine_body(path):
    html = path.read_text(encoding="utf-8")
    assert html.count(ENGINE_TITLE) == 1, "engine title text not found exactly once"
    html = html.replace(ENGINE_TITLE, NEW_TITLE)
    inner = html[html.index("<body>") + len("<body>"): html.rindex("</body>")]
    raw, escaped = "/" + chr(0xFFFD) + "/g", "/" + chr(92) + "ufffd/g"  # plotly.js strips U+FFFD with a literal regex; spell it as the JS escape
    assert inner.count(raw) == 1, "expected exactly one literal U+FFFD regex in plotly.js"
    inner = inner.replace(raw, escaped)
    assert "<script" in inner and "plotly-graph-div" in inner
    assert not re.search(r'<script[^>]+src="https?:', inner), "engine page loads an external script"
    return inner


def quantile_rows(fc):
    out = []
    for m in ("BOOT0", "BOOT", "EMP", "ENS"):
        q = fc["models"][m]["net_quantiles"]
        cells = "".join(f"<td class='num {'pos' if q[k] > 0 else 'neg'}'>{pct(q[k])}</td>" for k in ("05", "10", "25", "50", "75", "90", "95"))
        out.append(f"<tr><th scope='row'>{NAMES[m]}<span class='sub'>{WHAT[m]}</span></th>{cells}<td class='num'>{thb(10000 * (1 + q['50']))}</td></tr>")
    return "\n".join(out)


def band_rows(fc):
    bands = list(fc["models"]["BOOT0"]["bands"].keys())
    out = []
    for b in bands:
        cells = "".join(f"<td class='num'>{fc['models'][m]['bands'][b] * 100:.0f}%</td>" for m in ("BOOT0", "BOOT", "EMP", "ENS"))
        label = re.sub(r"-(\d)", MINUS + r"\1", b[0].upper() + b[1:])
        out.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    return "\n".join(out)


def dip_rows(fc):
    out = []
    for m in ("BOOT0", "BOOT", "RW", "FHS"):
        ps = fc["models"][m]["path_stats"]
        out.append(f"<tr><th scope='row'>{NAMES[m]}</th>" + "".join(f"<td class='num'>{ps[k] * 100:.0f}%</td>" for k in ("touch_down_15", "touch_down_25", "touch_down_35", "touch_up_20")) + "</tr>")
    return "\n".join(out)


def test_rows(fc):
    out = []
    for r in fc["walk_forward"]:
        m = r["model"]
        verdict = {"better": "Better than history", "worse": "Worse than history", "no clear difference": "Same as history"}[r["verdict"]]
        if m == "EMP":
            verdict = "The yardstick"
        band = r["80% band hit (want 80)"]
        band_class = "pos" if 72 <= band <= 88 else "neg"
        err = "-" if m == "EMP" else pct(r["vs EMP"])
        out.append(f"<tr><th scope='row'>{NAMES[m]}</th><td class='num {band_class}'>{band:.0f}%</td><td class='num'>{r['direction right (%)']:.0f}%</td>"
                   f"<td class='num {'neg' if r['verdict'] == 'worse' else ''}'>{err}</td><td>{verdict}</td></tr>")
    return "\n".join(out)


def build(fc, body):
    q0, q1, qe = (fc["models"][m]["net_quantiles"] for m in ("BOOT0", "BOOT", "EMP"))
    ps0, ps1 = fc["models"]["BOOT0"]["path_stats"], fc["models"]["BOOT"]["path_stats"]
    wf = {r["model"]: r for r in fc["walk_forward"]}
    losses = sorted(fc["models"][m]["P_loss"] for m in ("BOOT0", "BOOT", "EMP", "QLIN", "QGBM"))
    down_lo, down_hi = min(fc["models"][m]["P_loss"] for m in ("BOOT0", "BOOT", "EMP")), max(fc["models"][m]["P_loss"] for m in ("BOOT0", "BOOT", "EMP"))
    b = {m: fc["models"][m]["bands"] for m in ("BOOT0", "BOOT", "EMP")}
    band_key = list(b["BOOT0"].keys())
    last = fc["last_close"]
    subs = {
        "TODAY": day(fc["date"]), "LAST": thb(last),
        "MED0": pct(q0["50"]), "MED1": pct(q1["50"]), "P10_0": pct(q0["10"]), "P90_0": pct(q0["90"]), "P10_1": pct(q1["10"]), "P90_1": pct(q1["90"]),
        "DOWN_LO": f"{down_lo * 100:.0f}%", "DOWN_HI": f"{down_hi * 100:.0f}%",
        "T25_LO": f"{min(ps0['touch_down_25'], ps1['touch_down_25']) * 100:.0f}%", "T25_HI": f"{max(ps0['touch_down_25'], ps1['touch_down_25']) * 100:.0f}%",
        "T15_LO": f"{min(ps0['touch_down_15'], ps1['touch_down_15']) * 100:.0f}%", "T15_HI": f"{max(ps0['touch_down_15'], ps1['touch_down_15']) * 100:.0f}%",
        "T35_LO": f"{min(ps0['touch_down_35'], ps1['touch_down_35']) * 100:.0f}%", "T35_HI": f"{max(ps0['touch_down_35'], ps1['touch_down_35']) * 100:.0f}%",
        "THB_P10": thb(10000 * (1 + q0["10"])), "THB_MED": thb(10000 * (1 + q0["50"])), "THB_P90": thb(10000 * (1 + q0["90"])),
        "QUANTILES": quantile_rows(fc), "BANDS": band_rows(fc), "DIPS": dip_rows(fc), "TESTS": test_rows(fc), "BODY": body,
        "UP_SHARE": f"{wf['EMP']['windows really up (%)']:.0f}%", "DIR_LO": f"{min(r['direction right (%)'] for r in fc['walk_forward']):.0f}%", "DIR_HI": f"{max(r['direction right (%)'] for r in fc['walk_forward']):.0f}%",
        "BOOT_BAND": f"{wf['BOOT0']['80% band hit (want 80)']:.0f}%", "FHS_BAND": f"{wf['FHS']['80% band hit (want 80)']:.0f}%", "RW_BAND": f"{wf['RW']['80% band hit (want 80)']:.0f}%",
        "QLIN_LOSS": f"{wf['QLIN']['vs EMP'] * 100:.0f}%", "QGBM_LOSS": f"{wf['QGBM']['vs EMP'] * 100:.0f}%",
        "CRASH": f"{min(x[band_key[0]] for x in b.values()) * 100:.0f}% to {max(x[band_key[0]] for x in b.values()) * 100:.0f}%",
    }
    page = TEMPLATE
    for key, value in subs.items():
        page = page.replace(f"@@{key}@@", value)
    leftover = re.findall(r"@@[A-Z0-9_]+@@", page)
    assert not leftover, f"unfilled placeholders: {leftover}"
    return page


TEMPLATE = r"""<title>BTC 91-Day Outlook</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column; answer first (facts strip), then the range table, scenario shares, LumiBot's chart, dips on the way, the model test, how to read it. Dark to match LumiBot's chart. */
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
.sub { display: block; font: 400 12px/1.4 var(--mono); color: var(--muted); margin-top: 2px; max-width: 42ch; white-space: normal; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pos { color: var(--green); } .neg { color: var(--red); }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0; border-block: 1px solid var(--line); }
.fact { padding: 16px 20px 16px 0; display: flex; flex-direction: column; gap: 4px; }
.fact .label { font: 500 12px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.fact .value { font: 500 26px/1.2 var(--mono); font-variant-numeric: tabular-nums; }
.fact .note { font-size: 13px; color: var(--muted); }

.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; width: 100%; min-width: 680px; font-size: 15px; }
th, td { text-align: left; padding: 12px 16px 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { font: 500 12px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); border-bottom-color: var(--faint); }
tbody th { font-weight: 500; }

.plotscroll { overflow-x: auto; max-width: 100%; border: 1px solid var(--line); background: #111111; }
.plotframe { min-width: 980px; }
.guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }
.guide div { display: flex; flex-direction: column; gap: 6px; font-size: 15px; color: var(--muted); }
.guide b { font: 500 13px/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--fg); }
.read { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 28px 40px; }
.read p { color: var(--fg); }
.fine { font-size: 13px; color: var(--muted); max-width: 82ch; }
footer { display: flex; flex-direction: column; gap: 10px; border-top: 1px solid var(--line); padding-top: 24px; }
@media (max-width: 560px) { .fact { padding-right: 12px; } .fact .value { font-size: 22px; } }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Bitkub BTC/THB &middot; daily candles to @@TODAY@@ &middot; drawn with LumiBot</div>
  <h1>BTC 91-Day Outlook</h1>
  <p class="lede">If you buy BTC at @@LAST@@ THB today and sell in three months, the middle outcome is close to nothing: @@MED0@@ to @@MED1@@ after fees. The range is wide, from about @@P10_0@@ to @@P90_0@@ for most outcomes, and none of the seven models I tested can tell whether the price will go up or down.</p>
</header>

<div class="facts">
  <div class="fact"><span class="label">BTC today</span><span class="value">@@LAST@@</span><span class="note">THB, close of @@TODAY@@</span></div>
  <div class="fact"><span class="label">Middle outcome</span><span class="value">@@MED0@@ to @@MED1@@</span><span class="note">after 91 days and 0.25% fees each way</span></div>
  <div class="fact"><span class="label">Ends below your price</span><span class="value neg">@@DOWN_LO@@ to @@DOWN_HI@@</span><span class="note">chance, depending on the model</span></div>
  <div class="fact"><span class="label">Dips 25% on the way</span><span class="value neg">@@T25_LO@@ to @@T25_HI@@</span><span class="note">chance at some point in 91 days</span></div>
</div>

<section>
  <h2>The range after 91 days</h2>
  <p>Each row is one way of estimating the spread. The columns show the return after fees at the 5th, 10th and so on up to the 95th percentile: 10% of outcomes fall below the 10th-percentile figure and 10% rise above the 90th. The last column turns the middle outcome into 10,000 THB.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Model</th><th>5th</th><th>10th</th><th>25th</th><th>Middle</th><th>75th</th><th>90th</th><th>95th</th><th>10,000 THB becomes</th></tr></thead>
    <tbody>
@@QUANTILES@@
    </tbody>
  </table>
  </div>
  <p>With the zero-drift model, 10,000 THB ends between @@THB_P10@@ and @@THB_P90@@ in 8 cases out of 10, with a middle value of @@THB_MED@@. The two bootstrap rows are the ones to read, because they matched reality best in the test further down.</p>
</section>

<section>
  <h2>Five ways it could end</h2>
  <p>The same estimates, sorted into five outcome bands. Shares are of the simulated or historical outcomes after fees.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Outcome after 91 days</th><th>Bootstrap, no drift</th><th>Bootstrap, past drift</th><th>Plain history</th><th>Average of all seven</th></tr></thead>
    <tbody>
@@BANDS@@
    </tbody>
  </table>
  </div>
  <p class="fine">Shares come from past price moves. They describe how wide the range has been and carry no promise about the next three months.</p>
</section>

<section>
  <h2>The chart</h2>
  <div class="guide">
    <div><b>1 &middot; Price and the range ahead</b>The white line is BTC/THB with the 50-day (dotted blue) and 200-day (dashed orange) averages. Right of today, the red, grey and green lines are the bad, typical and good case (10th, 50th and 90th percentile) from the zero-drift bootstrap, with dashed lines for the bootstrap that keeps the past drift and dotted violet for the volatility simulation, which turned out too narrow. Yellow rings at day 91 mark the average of all seven models.</div>
    <div><b>2 &middot; Chance it has dipped by day N</b>How likely BTC is to have been 15%, 25% or 35% below today's price at some point by each day. These curves only rise, because a dip that has happened stays on the record.</div>
  </div>
  <div class="plotscroll"><div class="plotframe">
@@BODY@@
  </div></div>
  <p class="fine">Drag across a panel to zoom, use the buttons above each panel for 1 month to all, and click a legend entry to hide or show that line. Hover any point for the exact value.</p>
</section>

<section>
  <h2>Dips on the way</h2>
  <p>The price at day 91 is only part of the story. Each simulated path also has a lowest point, and that matters if you would sell after a fall. Bitkub has no stop orders, so any exit rule would have to be a sale at market after a check.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Model</th><th>Falls 15% at some point</th><th>Falls 25%</th><th>Falls 35%</th><th>Rises 20%</th></tr></thead>
    <tbody>
@@DIPS@@
    </tbody>
  </table>
  </div>
  <p>A 15% dip happens in @@T15_LO@@ to @@T15_HI@@ of paths and a 25% dip in @@T25_LO@@ to @@T25_HI@@, so a tight exit rule would very likely fire. A 35% dip is much rarer, @@T35_LO@@ to @@T35_HI@@. The random walk and volatility rows are lower because those models underestimated the width in the test below.</p>
</section>

<section>
  <h2>Which models can be trusted</h2>
  <p>Before using any model for today, I checked how it would have done in the past. Every 7th day since June 2020, each model made a 91-day forecast using only data up to that day, and I compared it with what happened next. That is 320 forecasts, about 25 independent quarters, since neighbouring days overlap.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Model</th><th>Outcomes inside its 10th-90th band (should be 80%)</th><th>Direction called right</th><th>Forecast error vs plain history</th><th>Result</th></tr></thead>
    <tbody>
@@TESTS@@
    </tbody>
  </table>
  </div>
  <div class="read">
    <p><b>Width.</b> The bootstrap models caught about @@BOOT_BAND@@ of outcomes in their 80% band, which is what an honest forecast should do. The random walk (@@RW_BAND@@) and the volatility simulation (@@FHS_BAND@@) caught far fewer. They were too confident because calm weeks do not last three months.</p>
    <p><b>Direction.</b> @@UP_SHARE@@ of the 91-day windows ended higher. No model called the direction right more often than that: scores ranged from @@DIR_LO@@ to @@DIR_HI@@. The forecasts are useful for how wide the range is, not for which way it will go.</p>
    <p><b>Machine learning.</b> Both machine-learning models made larger errors than plain history (@@QLIN_LOSS@@ and @@QGBM_LOSS@@ worse), which the test shows beyond chance. With only about 25 independent quarters, there is probably too little data for them to learn from.</p>
    <p><b>Caveat.</b> I picked the two bootstrap models as the ones to read after seeing this test, so their calibration looks a little better than it would on fresh data.</p>
  </div>
</section>

<section>
  <h2>How I read it</h2>
  <div class="read">
    <p>The outlook is a wide, roughly even spread around zero. About half the time you are down after three months, and about one time in four you see a 25% drop on the way. Neither the trend lines nor the models change that in any way I could measure.</p>
    <p>Size the position for the bad case, not the middle one. A loss of more than 30% after fees is the outcome in about @@CRASH@@ of cases over a single quarter, and a quarter is a short time in Bitcoin's history.</p>
    <p>Because Bitkub has no stop orders, a protective exit depends on a program or on you checking and selling by hand. Weeks when a computer is off or asleep are weeks without that protection.</p>
    <p>This page describes how uncertain the next three months are. It is not a buy or sell recommendation.</p>
  </div>
</section>

<footer>
  <p class="fine"><b>How it was made.</b> A LumiBot strategy ran over real Bitkub BTC/THB daily candles with the BitkubBacktesting data source. It reads the 50- and 200-day averages from <code>self.indicators</code>, draws with <code>add_line</code> and <code>add_marker</code> (dated into the future for the range), and LumiBot's <code>plot_indicators</code> wrote the chart above. The forecast maths (bootstrap, simulation and the two machine-learning models) is numpy and scikit-learn, because LumiBot has no tool for forecasting a 91-day distribution.</p>
  <p class="fine">Candle days are UTC. Returns include the Bitkub fee of 0.25% on the buy and on the sell. Percentages for the models that give only a range of values (plain history and machine learning) are read from that range, with the outer tails approximated. Past results do not guarantee future ones. Experiment E021.</p>
</footer>
</div>
"""


def main():
    out = Path(sys.argv[1])
    engine = max(LOGS.glob("Btc3mForecast_*_indicators.html"), key=lambda p: p.stat().st_mtime)
    fc = json.loads((LOGS / "btc_3m_forecast.json").read_text(encoding="utf-8"))
    page = build(fc, engine_body(engine))
    out.write_text(page, encoding="utf-8")
    print(f"engine chart: {engine.name}\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
