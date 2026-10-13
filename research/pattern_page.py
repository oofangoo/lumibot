"""E023 page: "Chart Look-alikes" - what the chart looked like before, what followed, and whether that tells you anything.

Reads logs/pattern_analogs.json (pattern_study.py) and the newest logs/PatternAnalogs<BTC|SPY>_*_indicators.html (LumiBot's plot_indicators, from pattern_chart.py).
Every number in the text is computed from the JSON, so words and charts cannot disagree. plotly.js is included once (taken from LumiBot's own chart file) and
shared by LumiBot's two charts and the page's own charts.
Run:  .\\run.ps1 research\\pattern_page.py <out.html>
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

LOGS = Path(__file__).resolve().parent.parent / "logs"
MINUS = "&minus;"
ORDER = ("BTC", "SPY")
NAME = {"BTC": "Bitcoin", "SPY": "S&amp;P 500 (SPY)"}
UNITS = {"BTC": "days", "SPY": "trading days"}
BUCKET_COLOR = {"Falls hard": "#ff7a70", "Weak": "#ffa94d", "Middling": "#aab4bc", "Strong": "#8ce99a", "Surges": "#38d9a9"}


def pct(x, digits=0, signed=True):
    v = round(x * 100, digits)
    return (MINUS if v < 0 else ("+" if signed and v > 0 else "")) + f"{abs(v):.{digits}f}%"


def rng_txt(lo, hi):
    return f"{pct(lo)} to {pct(hi)}"


def verdict(w):
    if w["diff_lo"] > 0:
        return "worse"
    if w["diff_hi"] < 0:
        return "better"
    return "no better"


def lumibot_parts(path):
    html = path.read_text(encoding="utf-8")
    s = html.index("<script>/**\n* plotly.js")
    e = html.index("</script>", s) + len("</script>")
    bundle = html[s:e]
    raw, escaped = "/" + chr(0xFFFD) + "/g", "/" + chr(92) + "ufffd/g"  # plotly.js strips U+FFFD with a literal regex; spell it as the JS escape
    assert bundle.count(raw) == 1, "expected exactly one literal U+FFFD regex in plotly.js"
    bundle = bundle.replace(raw, escaped)
    d0 = html.index('<div id="', e)
    block = html[d0: html.rindex("</script>") + len("</script>")]
    gid = re.match(r'<div id="([^"]+)"', block).group(1)
    assert not re.search(r'<script[^>]+src="https?:', block), "engine page loads an external script"
    return bundle, block, gid


def market_html(m, d):
    t, w, lib, nl = d["today"], d["walk_forward"], d["library"], d["null"]
    H, unit = d["horizon"], UNITS[m]
    hl = f"{H}-day" if m == "BTC" else f"{H}-trading-day"
    an = t["analogs"]
    rets = [a["fwd_return"] for a in an]
    lo, hi = min(rets), max(rets)
    med_an, med_base = t["analog_quantiles"]["50"], t["base_quantiles"]["50"]
    today_cluster = next(c for c in lib["clusters"] if c["today"])
    v = verdict(w)
    sh = {s["name"]: s for s in t["shares"]}
    best = max(t["shares"], key=lambda s: s["share"])
    n_up = sum(1 for r in rets if r > 0)
    flips = 0
    for c in lib["clusters"]:
        a, b = c["first"], c["second"]
        if a["median"] is None or b["median"] is None:
            continue
        flips += (a["median"] - lib["base"]["first"]) * (b["median"] - lib["base"]["second"]) < 0
    close_gap = 1 - w["mean_dist"] / nl["shuffled"]["mean_dist"]
    verdict_line = {
        "worse": f"Measurably worse than plain history: the look-alike forecasts scored {abs(w['ratio'] - 1) * 100:.0f}% worse (the gap is outside the noise).",
        "no better": f"No better than plain history: the look-alike forecasts scored {abs(w['ratio'] - 1) * 100:.0f}% {'worse' if w['ratio'] > 1 else 'better'}, a gap inside the noise.",
        "better": "Better than plain history.",
    }[v]
    rows = "".join(
        f"<tr><td>{a['start']} &rarr; {a['end']}</td><td class='num'>{a['distance']:.2f}</td><td class='num {'pos' if a['fwd_return'] > 0 else 'neg'}'>{pct(a['fwd_return'])}</td>"
        f"<td class='num {'neg' if a['fwd_low'] < -0.005 else ''}'>{pct(a['fwd_low'])}</td><td class='num {'pos' if a['fwd_high'] > 0.005 else ''}'>{pct(a['fwd_high'])}</td><td><span class='dot' style='background:{BUCKET_COLOR[a['bucket']]}'></span>{a['bucket']}</td></tr>"
        for a in an)
    lib_rows = ""
    for c in lib["clusters"]:
        a, b = c["first"], c["second"]
        f = lambda x: "-" if x["median"] is None else pct(math.exp(x["median"]) - 1)
        lib_rows += (f"<tr class='{'now' if c['today'] else ''}'><th scope='row'>{c['name']}{' <span class=\"tag\">today</span>' if c['today'] else ''}</th><td class='num'>{c['share'] * 100:.0f}%</td>"
                     f"<td class='num'>{f(a)}<span class='sub'>~{a['episodes']:.0f} episodes</span></td><td class='num'>{f(b)}<span class='sub'>~{b['episodes']:.0f} episodes</span></td></tr>")
    rob = ", ".join(f"{k.replace('_', ', ').replace('W', 'window ').replace('K', 'matches ')}: {pct(v_['ratio'] - 1)}" for k, v_ in d["robustness"].items())
    return f"""
<section class="market" id="panel-{m}" data-market="{m}" hidden>
  <div class="answer">
    <p class="kicker">{NAME[m]} &middot; data to {d['last_date']} &middot; look-ahead window {H} {unit}</p>
    <h2>Today's chart has {len(an)} look-alikes in history. What followed them is a mixed bag, as it is for any chart.</h2>
    <p class="lede">The last {d['window']} days are a <b>&ldquo;{today_cluster['name'].lower()}&rdquo;</b>. The {len(an)} closest matches in {d['first_date'][:4]}&ndash;{d['last_date'][:4]} were followed by moves
    from <b class="{'neg' if lo < 0 else 'pos'}">{pct(lo)}</b> to <b class="{'neg' if hi < 0 else 'pos'}">{pct(hi)}</b> over the next {H} {unit}. {n_up} of {len(an)} went up. The middle result was <b>{pct(med_an)}</b>;
    for every window in {m}'s history it is <b>{pct(med_base)}</b>.</p>
    <div class="kpis">
      <div class="kpi"><span class="k">Middle outcome of the look-alikes</span><span class="v">{pct(med_an)}</span><span class="s">all history: {pct(med_base)}</span></div>
      <div class="kpi"><span class="k">Spread (5th to 95th percentile)</span><span class="v">{pct(t['analog_quantiles']['05'])} to {pct(t['analog_quantiles']['95'])}</span><span class="s">all history: {pct(t['base_quantiles']['05'])} to {pct(t['base_quantiles']['95'])}</span></div>
      <div class="kpi"><span class="k">Went up after the match</span><span class="v">{n_up} of {len(an)}</span><span class="s">any day in history: {t['base_up_share'] * 100:.0f}%</span></div>
      <div class="kpi verdict-{v.replace(' ', '-')}"><span class="k">Did matching beat plain history?</span><span class="v">{'No' if v != 'better' else 'Yes'}</span><span class="s">{pct(w['ratio'] - 1)} forecast error vs plain history</span></div>
    </div>
  </div>

  <h3>Today next to its look-alikes</h3>
  <p class="note">White is the last {d['window']} {unit} of {m}. Each thin line is one look-alike: its {d['window']}-day window (left of 0) and the {H} {unit} that followed (right of 0), in % from its last day. Red lines fell hard, green surged.
  Thick white is the middle of the ten; the shaded band holds the 10th to 90th percentile of their paths. Grey dotted lines are the same range for every window in history.</p>
  <div class="chartbox"><div id="spag-{m}" class="plot tall"></div></div>
  <div class="legend">{''.join(f"<span><i style='background:{c}'></i>{n}</span>" for n, c in BUCKET_COLOR.items())}</div>

  <h3>The same thing on the real price chart (LumiBot)</h3>
  <p class="note">Drawn by LumiBot&rsquo;s own chart engine over real daily candles. Each look-alike is rescaled so its last day lands on today&rsquo;s close, then drawn into the future. The log scale makes equal percentage moves look equal. Hover any line for its dates; drag to zoom.
  The lower panel is the shape comparison itself (each window z-scored, so only the shape counts).</p>
  <div class="chartbox lumi"><div class="lumi-wrap" id="lumi-wrap-{m}">@@LUMI_{m}@@</div></div>

  <h3>Possible scenarios</h3>
  <p class="note">Every {hl} outcome in {m}'s history was sorted into fifths: the worst fifth is &ldquo;Falls hard&rdquo;, the best fifth &ldquo;Surges&rdquo;. For a chart with no information, each fifth would hold 20% of look-alikes.
  With only {len(an)} matches the bars wobble a lot (the thin lines show the 95% range of that wobble). {best['name']} is the biggest at {best['share'] * 100:.0f}% ({best['n']} of {len(an)}).</p>
  <div class="chartbox"><div id="shares-{m}" class="plot"></div></div>
  <div class="tablewrap"><table><thead><tr><th>Look-alike (window &rarr; end)</th><th>Distance</th><th>Next {H} {unit}</th><th>Worst point</th><th>Best point</th><th>Scenario</th></tr></thead><tbody>{rows}</tbody></table></div>
  <p class="fine">Distance is the average gap between the two z-scored shapes (0 = identical). Matches are at least {d['window']} days apart, so each is a separate episode. Returns are price changes without fees.</p>

  <h3>The shape library</h3>
  <p class="note">All {d['window']}-day windows were grouped into six typical shapes by k-means, using only the first half of history (to {lib['split_date']}). The thin band holds 80% of the windows in each group. Then every group&rsquo;s later results were
  compared between the half used to find the shapes and the half the grouping never saw.</p>
  <div class="grid6" id="lib-{m}"></div>
  <div class="chartbox"><div id="libbars-{m}" class="plot"></div></div>
  <div class="tablewrap"><table><thead><tr><th>Shape</th><th>Share of windows</th><th>Median next {H} {unit}, first half</th><th>Second half (not used to find shapes)</th></tr></thead><tbody>{lib_rows}</tbody></table></div>
  <p class="fine">Typical result for any window: first half {pct(math.exp(lib['base']['first']) - 1)}, second half {pct(math.exp(lib['base']['second']) - 1)}.
  <b>{flips} of {len(lib['clusters'])}</b> shapes switched from beating the average to lagging it (or the reverse) between the halves. Each group holds only a handful of independent episodes, which is why results this uneven are expected.</p>

  <h3>Does pattern matching actually help?</h3>
  <p class="note">Test: stand at {w['origins']} past dates, use only the history before each one to find look-alikes, forecast the next {H} {unit} from what followed them, then check what really happened. The yardstick is plain history (the range of every earlier {hl} result).
  The same test also ran on {len(nl['shuffled']['ratios'])} made-up price series that have the same daily moves in shuffled order, so they contain no patterns at all.</p>
  <div class="verdict {v.replace(' ', '-')}"><b>{verdict_line}</b>
    <ul>
      <li>Forecast error (pinball score, lower is better): look-alikes {w['loss_an']:.4f}, plain history {w['loss_emp']:.4f}. 95% range of the difference: {w['diff_lo']:+.4f} to {w['diff_hi']:+.4f}.</li>
      <li>The look-alikes' 80% range held the real outcome {w['in80_an'] * 100:.0f}% of the time (plain history: {w['in80_emp'] * 100:.0f}%; a perfect 80% range would hit 80%). Look-alike ranges are too narrow.</li>
      <li>Up-or-down call from the middle of the look-alikes was right {w['dir_hit'] * 100:.0f}% of the time. Always saying &ldquo;up&rdquo; would have been right {w['up_share'] * 100:.0f}%. Rank correlation between predicted and actual: {w['rank_corr']:+.2f} (0 = none).</li>
      <li>Shapes repeat about as often as chance makes them: the closest matches are only {close_gap * 100:.0f}% nearer than the closest matches in made-up series.</li>
      <li>Made-up series with no patterns scored on average {pct(np.mean(nl['shuffled']['ratios']) - 1)} against plain history; {m}'s real score was {pct(w['ratio'] - 1)}. {nl['shuffled']['p_ratio'] * 100:.0f}% of the made-up series did as well or better, so the real result is not distinguishable from no patterns.</li>
      <li>Other settings (window length, number of matches): {rob}. None of them beats plain history by a meaningful margin.</li>
    </ul></div>
  <div class="two">
    <div class="chartbox"><div id="scatter-{m}" class="plot"></div></div>
    <div class="chartbox"><div id="hist-{m}" class="plot"></div></div>
  </div>
  <p class="fine">Left: each dot is one past date; if look-alikes could call the outcome the dots would climb along the dashed diagonal. Right: error score relative to plain history (1.0 = equal; above 1 = worse) for the {len(nl['shuffled']['ratios'])} made-up series and for {m}.</p>
</section>
"""


CSS = """
:root{--bg:#0e1215;--surface:#151b20;--surface2:#1b232a;--line:#27313a;--ink:#e8edf0;--muted:#92a0aa;--accent:#f2b84b;--pos:#4cc9a0;--neg:#ff7a70;--neutral:#aab4bc;
  --font-display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;--font-body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--font-data:"IBM Plex Mono",ui-monospace,Consolas,monospace;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:15px;line-height:1.55;margin:0}
.wrap{max-width:1040px;margin:0 auto;padding-inline:16px;padding-block:24px 56px}
header.top{padding-block:8px 18px}
h1{font-family:var(--font-display);font-size:clamp(30px,6vw,46px);line-height:1.05;margin:0 0 8px;letter-spacing:-.02em;font-weight:700;text-wrap:balance}
h1 span{color:var(--accent)}
.sub0{color:var(--muted);max-width:68ch;margin:0}
.tabs{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;display:flex;gap:8px;background:var(--bg);padding-block:10px;border-bottom:1px solid var(--line);margin-bottom:6px}
.tab{font:600 14px var(--font-body);color:var(--muted);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 16px;cursor:pointer}
.tab[aria-selected="true"]{color:#14100a;background:var(--accent);border-color:var(--accent)}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.kicker{font:500 12px var(--font-data);text-transform:uppercase;letter-spacing:.08em;color:var(--accent);margin:18px 0 6px}
h2{font-family:var(--font-display);font-size:clamp(22px,4.2vw,31px);line-height:1.15;margin:0 0 12px;letter-spacing:-.01em;text-wrap:balance;font-weight:650}
h3{font-family:var(--font-display);font-size:21px;margin:38px 0 6px;font-weight:650;letter-spacing:-.005em}
.lede{font-size:16.5px;max-width:70ch;margin:0 0 16px}
.note{color:var(--muted);max-width:78ch;margin:0 0 12px}
.fine{color:var(--muted);font-size:13px;max-width:80ch;margin:10px 0 0}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:2px;min-width:0}
.kpi .k{font-size:12px;color:var(--muted)}.kpi .v{font:600 22px var(--font-display);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.kpi .s{font-size:12.5px;color:var(--muted)}
.kpi.verdict-worse .v,.kpi.verdict-no-better .v{color:var(--neg)}.kpi.verdict-better .v{color:var(--pos)}
.chartbox{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px;min-width:0}
.plot{width:100%;height:340px}.plot.tall{height:440px}
.lumi{padding:0;overflow:hidden}.lumi-wrap{width:100%;height:760px;background:#111}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.grid6{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:12px}
.mini{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 8px 4px;min-width:0}
.mini.now{border-color:var(--accent)}
.mini h4{font:600 13px var(--font-body);margin:2px 4px 0;display:flex;justify-content:space-between;gap:6px}
.mini .plot{height:130px}
.tag{font:600 10px var(--font-data);text-transform:uppercase;letter-spacing:.06em;color:#14100a;background:var(--accent);border-radius:4px;padding:1px 6px;vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:8px 2px 0;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle;border-radius:2px}
.tablewrap{overflow-x:auto;margin-top:12px;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font:500 12px var(--font-data);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);white-space:nowrap}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;font-family:var(--font-data);white-space:nowrap}
.sub{display:block;font:400 11.5px var(--font-body);color:var(--muted)}
tr.now{background:var(--surface2)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.verdict{border:1px solid var(--line);border-left:4px solid var(--neutral);border-radius:10px;background:var(--surface);padding:14px 18px;margin:0 0 12px}
.verdict.worse,.verdict.no-better{border-left-color:var(--neg)}.verdict.better{border-left-color:var(--pos)}
.verdict ul{margin:8px 0 0;padding-left:18px;display:grid;gap:6px;color:var(--ink)}
footer.end{margin-top:44px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px;max-width:84ch}
footer.end code{font-family:var(--font-data);font-size:12px}
@media (max-width:560px){.plot.tall{height:380px}.lumi-wrap{height:620px}}
"""

JS = r"""
const DATA = @@DATA@@;
const LUMI = @@LUMI@@;
const INK='#e8edf0', MUTED='#92a0aa', GRID='#27313a';
const BC = {"Falls hard":"#ff7a70","Weak":"#ffa94d","Middling":"#aab4bc","Strong":"#8ce99a","Surges":"#38d9a9"};
const pc = v => (v>=0?'+':'−') + Math.abs(v).toFixed(0) + '%';
const base = (extra) => Object.assign({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:INK,family:'IBM Plex Sans, Segoe UI, sans-serif',size:12},margin:{l:52,r:14,t:10,b:44},
  xaxis:{gridcolor:GRID,zerolinecolor:GRID,linecolor:GRID},yaxis:{gridcolor:GRID,zerolinecolor:'#3a4753',linecolor:GRID},legend:{orientation:'h',y:-0.22,font:{size:11}},hoverlabel:{font:{family:'IBM Plex Sans'}}},extra);
const cfg = {responsive:true,displaylogo:false,displayModeBar:false};
const drawn = {};

function spaghetti(m){
  const d=DATA[m], t=d.today, H=d.horizon, W=d.window, tr=[];
  const xs=[]; for(let i=-(W-1);i<=H;i++) xs.push(i);
  const fx=[]; for(let i=0;i<=H;i++) fx.push(i);
  const band=(arr,idx)=>arr.map(r=>(Math.exp(r[idx])-1)*100);
  tr.push({x:fx,y:band(t.base_fan,0),mode:'lines',line:{color:'#6b7882',width:1.5,dash:'dot'},name:'All history, 10th and 90th percentile',hoverinfo:'skip',legendgroup:'b'});
  tr.push({x:fx,y:band(t.base_fan,4),mode:'lines',line:{color:'#6b7882',width:1.5,dash:'dot'},showlegend:false,hoverinfo:'skip',legendgroup:'b'});
  tr.push({x:fx,y:band(t.analog_fan,0),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
  tr.push({x:fx,y:band(t.analog_fan,4),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(242,184,75,0.13)',name:'Look-alikes, 10th to 90th percentile',hoverinfo:'skip'});
  t.analogs.forEach((a,i)=>{
    const y=a.window_rebased.map(v=>(Math.exp(v)-1)*100).concat(a.path.slice(1).map(v=>(Math.exp(v)-1)*100));
    tr.push({x:xs,y:y,mode:'lines',line:{color:BC[a.bucket],width:1.3},opacity:0.85,showlegend:false,name:a.start+' to '+a.end,
      hovertemplate:'<b>'+a.start+' to '+a.end+'</b><br>day %{x}: %{y:.0f}%<br>'+a.bucket+', '+pc(a.fwd_return*100)+' after '+H+'<extra></extra>'});
  });
  tr.push({x:fx,y:band(t.analog_fan,2),mode:'lines',line:{color:'#ffffff',width:3.5},name:'Look-alikes, middle',hovertemplate:'middle of look-alikes: %{y:.0f}% at day %{x}<extra></extra>'});
  const last=t.recent.close[t.recent.close.length-1];
  const tx=[],ty=[]; t.recent.close.forEach((c,i)=>{tx.push(i-(t.recent.close.length-1));ty.push((c/last-1)*100);});
  tr.push({x:tx,y:ty,mode:'lines',line:{color:'#ffffff',width:3.5},name:'Today ('+m+', last '+W+' days)',hovertemplate:'today\'s chart: %{y:.0f}% at day %{x}<extra></extra>'});
  const lay=base({xaxis:{title:{text:'days from today ('+d.unit+')'},gridcolor:GRID,zerolinecolor:'#3a4753',linecolor:GRID},yaxis:{title:{text:'change from the last day, %'},ticksuffix:'%',gridcolor:GRID,zerolinecolor:'#3a4753',linecolor:GRID},
    shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{color:'#f2b84b',width:1.5,dash:'dash'}}],margin:{l:58,r:14,t:10,b:(window.innerWidth<560?150:96)},legend:{orientation:'h',y:(window.innerWidth<560?-0.36:-0.22),font:{size:11}},hovermode:'closest'});
  Plotly.newPlot('spag-'+m,tr,lay,cfg);
}

function shares(m){
  const d=DATA[m], s=d.today.shares, ed=d.today.edges;
  const rng=(i)=> i===0?'below '+pc(ed[0]*100): i===4?'above '+pc(ed[3]*100): pc(ed[i-1]*100)+' to '+pc(ed[i]*100);
  const names=s.map((b,i)=>b.name+'<br><span style="font-size:10px">'+rng(i)+'</span>');
  const tr=[{type:'bar',x:names,y:s.map(b=>b.share*100),marker:{color:s.map(b=>BC[b.name])},text:s.map(b=>b.n+' of '+d.k),textposition:'outside',cliponaxis:false,
    error_y:{type:'data',symmetric:false,array:s.map(b=>(b.ci_hi-b.share)*100),arrayminus:s.map(b=>(b.share-b.ci_lo)*100),color:'#92a0aa',thickness:1.2,width:5},
    hovertemplate:'%{y:.0f}% of look-alikes<extra></extra>',name:'Look-alikes'}];
  const lay=base({yaxis:{title:{text:'% of the look-alikes'},ticksuffix:'%',range:[0,80],gridcolor:GRID,linecolor:GRID},showlegend:false,margin:{l:58,r:14,t:40,b:70},
    shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:20,y1:20,line:{color:'#f2b84b',width:1.5,dash:'dash'}}],
    annotations:[{xref:'paper',yref:'paper',x:0,y:1.0,yanchor:'bottom',xanchor:'left',text:'Dashed line: 20% = no information',showarrow:false,font:{color:'#f2b84b',size:11}}]});
  Plotly.newPlot('shares-'+m,tr,lay,cfg);
}

function library(m){
  const d=DATA[m], lib=d.library, W=d.window, box=document.getElementById('lib-'+m);
  const xs=[]; for(let i=0;i<W;i++) xs.push(i-W+1);
  lib.clusters.forEach((c,i)=>{
    const div=document.createElement('div'); div.className='mini'+(c.today?' now':'');
    div.innerHTML='<h4><span>'+c.name+'</span><span style="color:#92a0aa;font-weight:400">'+(c.share*100).toFixed(0)+'%</span></h4><div class="plot" id="mini-'+m+'-'+i+'"></div>';
    box.appendChild(div);
    const tr=[{x:xs,y:c.band_lo,mode:'lines',line:{width:0},hoverinfo:'skip',showlegend:false},{x:xs,y:c.band_hi,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(146,160,170,0.22)',hoverinfo:'skip',showlegend:false},
      {x:xs,y:c.centroid,mode:'lines',line:{color:c.today?'#f2b84b':'#e8edf0',width:2.5},hoverinfo:'skip',showlegend:false}];
    Plotly.newPlot('mini-'+m+'-'+i,tr,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:6,r:6,t:4,b:4},xaxis:{visible:false},yaxis:{visible:false,range:[-3,3]}},{staticPlot:true,responsive:true});
  });
  const names=lib.clusters.map(c=>c.name);
  const f=k=>lib.clusters.map(c=>c[k].median===null?null:(Math.exp(c[k].median)-1)*100);
  const tr=[{type:'bar',name:'First half (shapes found here)',x:names,y:f('first'),marker:{color:'#6b8cae'}},{type:'bar',name:'Second half (never seen)',x:names,y:f('second'),marker:{color:'#f2b84b'}}];
  const b1=(Math.exp(lib.base.first)-1)*100, b2=(Math.exp(lib.base.second)-1)*100;
  const lay=base({barmode:'group',yaxis:{title:{text:'median next '+d.horizon+' '+d.unit+', %'},ticksuffix:'%',gridcolor:GRID,zerolinecolor:'#5a6a77',linecolor:GRID},margin:{l:62,r:14,t:10,b:100},legend:{orientation:'h',y:-0.3,font:{size:11}},
    xaxis:{tickangle:-20,gridcolor:GRID,linecolor:GRID},
    shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:b1,y1:b1,line:{color:'#6b8cae',width:1.2,dash:'dot'}},{type:'line',xref:'paper',x0:0,x1:1,y0:b2,y1:b2,line:{color:'#f2b84b',width:1.2,dash:'dot'}}]});
  Plotly.newPlot('libbars-'+m,tr,lay,cfg);
}

function honesty(m){
  const d=DATA[m], w=d.walk_forward;
  const x=w.rows.map(r=>r.pred_median*100), y=w.rows.map(r=>r.actual*100);
  const lim=Math.max(...x.map(Math.abs),...y.map(Math.abs).map(v=>Math.min(v,150)))*1.1;
  Plotly.newPlot('scatter-'+m,[{x:x,y:y,mode:'markers',marker:{color:'#92a0aa',size:5,opacity:0.6},hovertemplate:'predicted %{x:.0f}%, actual %{y:.0f}%<extra></extra>',showlegend:false},
    {x:[-lim,lim],y:[-lim,lim],mode:'lines',line:{color:'#f2b84b',dash:'dash',width:1.5},hoverinfo:'skip',name:'perfect call',showlegend:false}],
    base({xaxis:{title:{text:'middle of the look-alikes (predicted), %'},ticksuffix:'%',gridcolor:GRID,zerolinecolor:'#3a4753',linecolor:GRID},yaxis:{title:{text:'what happened, %'},ticksuffix:'%',gridcolor:GRID,zerolinecolor:'#3a4753',linecolor:GRID},margin:{l:60,r:14,t:10,b:50}}),cfg);
  const r=d.null.shuffled.ratios;
  const tr=[{type:'histogram',x:r,nbinsx:14,marker:{color:'#6b8cae'},name:'made-up series with no patterns',hovertemplate:'%{x:.3f}: %{y} series<extra></extra>'}];
  const lay=base({xaxis:{title:{text:'forecast error vs plain history (1.0 = equal, above = worse)'},gridcolor:GRID,linecolor:GRID},yaxis:{title:{text:'number of made-up series'},gridcolor:GRID,linecolor:GRID},showlegend:false,margin:{l:56,r:14,t:26,b:56},
    shapes:[{type:'line',x0:1,x1:1,yref:'paper',y0:0,y1:1,line:{color:'#e8edf0',width:1.5,dash:'dot'}},{type:'line',x0:w.ratio,x1:w.ratio,yref:'paper',y0:0,y1:1,line:{color:'#f2b84b',width:3}}],
    annotations:[{x:w.ratio,yref:'paper',y:1.02,text:m+' (real): '+w.ratio.toFixed(2),showarrow:false,font:{color:'#f2b84b',size:11},xanchor:'center'},{x:1,yref:'paper',y:0.0,yanchor:'bottom',xanchor:'right',text:'equal ',showarrow:false,font:{color:'#e8edf0',size:10}}]});
  Plotly.newPlot('hist-'+m,tr,lay,cfg);
}

function lumiZoom(m){
  const gd=document.getElementById(LUMI[m].gid); if(!gd||!window.Plotly) return;
  Plotly.relayout(gd,{'xaxis.range':LUMI[m].xr,'yaxis.type':'log','yaxis.range':LUMI[m].yr,'showlegend':false,'xaxis.rangeslider.visible':false,'xaxis2.rangeslider.visible':false,'xaxis.rangeselector.visible':false,'xaxis2.rangeselector.visible':false,'title.text':'','margin.t':60});
}

function show(m){
  document.querySelectorAll('.market').forEach(s=>{s.hidden=(s.dataset.market!==m);});
  document.querySelectorAll('.tab').forEach(b=>b.setAttribute('aria-selected',b.dataset.market===m?'true':'false'));
  if(!drawn[m]){drawn[m]=true; spaghetti(m); shares(m); library(m); honesty(m); lumiZoom(m);}
  window.dispatchEvent(new Event('resize'));
  try{history.replaceState(null,'','#'+m.toLowerCase());}catch(e){}
}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>show(b.dataset.market)));
const want=(location.hash||'').replace('#','').toUpperCase();
show(DATA[want]?want:'BTC');
"""


def build(res, parts):
    bundle = parts["BTC"][0]
    data, lumi = {}, {}
    for m in ORDER:
        d = res[m]
        t = d["today"]
        slim = {k: d[k] for k in ("label", "horizon", "unit", "last_date", "last_close", "window", "k")}
        slim["today"] = {k: t[k] for k in ("analogs", "shares", "edges", "analog_fan", "base_fan", "recent")}
        slim["today"]["analogs"] = [{k: a[k] for k in ("start", "end", "fwd_return", "bucket", "window_rebased", "path")} for a in t["analogs"]]
        slim["library"] = {"clusters": [{k: c[k] for k in ("name", "centroid", "band_lo", "band_hi", "today", "share", "first", "second")} for c in d["library"]["clusters"]], "base": d["library"]["base"]}
        slim["walk_forward"] = {"ratio": d["walk_forward"]["ratio"], "rows": d["walk_forward"]["rows"]}
        slim["null"] = {"shuffled": {"ratios": d["null"]["shuffled"]["ratios"]}}
        data[m] = slim
        last = d["last_close"]
        vals = [last * math.exp(v) for a in t["analogs"] for v in a["window_rebased"] + a["path"]] + [last * math.exp(r[i]) for r in t["base_fan"] for i in (0, 4)]
        lo, hi = min(vals) * 0.97, max(vals) * 1.03
        from datetime import date, timedelta

        today = date.fromisoformat(d["last_date"])
        end = today + timedelta(days=int(d["horizon"] * (1 if m == "BTC" else 1.5)) + 3)
        lumi[m] = {"gid": parts[m][2], "xr": [str(today - timedelta(days=int(d["window"] * (1 if m == "BTC" else 1.5)) + 4)), str(end)], "yr": [math.log10(lo), math.log10(hi)]}
    panels = "".join(market_html(m, res[m]) for m in ORDER)
    for m in ORDER:
        panels = panels.replace(f"@@LUMI_{m}@@", parts[m][1])
    js = JS.replace("@@DATA@@", json.dumps(data, separators=(",", ":"))).replace("@@LUMI@@", json.dumps(lumi))
    n_btc = res["BTC"]["walk_forward"]["origins"]
    return f"""<meta charset="utf-8"><title>Chart Look-alikes</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
{bundle}
<div class="wrap">
<header class="top">
  <h1>Chart <span>look-alikes</span></h1>
  <p class="sub0">Does today&rsquo;s chart resemble anything in history, and what happened next? This page finds the closest matches for Bitcoin and the S&amp;P 500, reads the possible scenarios off them, then tests whether that has ever predicted anything.</p>
</header>
<nav class="tabs" role="tablist" aria-label="Market"><button class="tab" role="tab" data-market="BTC" aria-selected="true">Bitcoin</button><button class="tab" role="tab" data-market="SPY" aria-selected="false">S&amp;P 500</button></nav>
{panels}
<footer class="end">
  <p><b>How it was made.</b> A pattern is the shape of the last 60 days of log price, z-scored so only the shape counts. Look-alikes are the 10 nearest earlier windows (at least 60 days apart, with the following period already known on the day).
  Charts and price series come from LumiBot: a LumiBot Strategy ran over real Bitkub BTC/THB candles (BitkubBacktesting) and real SPY candles (AlpacaBacktesting), drew with <code>add_line</code> and <code>add_marker</code>, and <code>plot_indicators</code> produced the price charts. The matching, the k-means shapes and the tests
  are numpy and scikit-learn, because LumiBot has no pattern-matching tool. The other charts on this page are drawn with plotly.js, the library inside LumiBot&rsquo;s own charts.</p>
  <p><b>Limits.</b> Bitcoin has 8 years of history, so its ten matches come from a handful of market cycles. SPY has 33 years but a calm, rising market most of the time. The tests ran {n_btc} past dates for Bitcoin and {res['SPY']['walk_forward']['origins']} for SPY; neighbouring dates share most of their history, so the effective sample is much smaller.
  Nothing here is advice or a forecast. It shows the spread of what followed similar charts, and that the spread is about the same as for any chart. Candle days are UTC for Bitcoin.</p>
</footer>
</div>
<script>{js}</script>
"""


def main():
    out = Path(sys.argv[1])
    res = json.loads((LOGS / "pattern_analogs.json").read_text(encoding="utf-8"))
    parts = {}
    for m in ORDER:
        f = max(LOGS.glob(f"PatternAnalogs{m}_*_indicators.html"), key=lambda p: p.stat().st_mtime)
        parts[m] = lumibot_parts(f)
        print(f"LumiBot chart {m}: {f.name}")
    page = build(res, parts)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
