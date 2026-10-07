"""Render the game model as one page: a weekly board plus a breakdown for every game.

    python scripts/game_pages.py            # current week and the one before -> site/nfl_board.html
    python scripts/game_pages.py 2026 4 5   # chosen season and weeks

Every number comes from game_model.py; nothing is entered by hand. The file is written without
<html>/<head>/<body> so it can be published as is; browsers open it fine locally too.
"""
import datetime as dt
import html
import sys
from pathlib import Path

import polars as pl

import game_model as gm
from props_week import TEAMS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "nfl_board.html"
NAME = {v: k for k, v in TEAMS.items()}
NAME.update({"LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams"})
NICK = {k: v.split()[-1] for k, v in NAME.items()}
BIG_GAP = 3.0
# percentile rows: (label, offense column, defense column, higher-is-better on defense?)
PCT_ROWS = [("Passing", "off_pass", "def_pass"), ("Rushing", "off_run", "def_run"),
            ("Pass consistency", "off_pass_sr", "def_pass_sr"), ("Rush consistency", "off_run_sr", "def_run_sr"),
            ("Pass rate", "off_pass_rate", "def_pass_rate"), ("Pace", "off_plays", "def_plays")]
# rows where a defence allowing less is better (EPA and success rate)
DEF_LOWER_BETTER = {"def_pass", "def_run", "def_pass_sr", "def_run_sr"}
RATING_ROWS = [("Pass EPA / dropback", "pass", "{:+.3f}"), ("Rush EPA / carry", "run", "{:+.3f}"),
               ("Pass success rate", "pass_sr", "{:+.3f}"), ("Rush success rate", "run_sr", "{:+.3f}"),
               ("Pass rate", "pass_rate", "{:+.3f}"), ("Plays per game", "plays", "{:+.1f}")]
e = html.escape


def clock(t: str | None) -> str:
    """'20:15' -> '8:15 PM ET'."""
    if not t:
        return ""
    h, m = map(int, t.split(":"))
    return f"{(h - 1) % 12 + 1}:{m:02d} {'PM' if h >= 12 else 'AM'} ET"


def spread_txt(team: str, margin: float) -> str:
    """A home-minus-away margin as 'TEAM -x' for the favourite."""
    if margin is None:
        return "no line yet"
    if abs(margin) < 0.05:
        return "Pick'em"
    return f"{team} -{abs(margin):.1f}"


def fav(g: dict, margin: float) -> str:
    return spread_txt(g["team"] if margin > 0 else g["a_team"], margin)


def records(sched: pl.DataFrame) -> dict:
    """(season, team) -> list of (gameday, W/L/T) for completed regular-season games."""
    out = {}
    for r in sched.filter(pl.col("home_score").is_not_null() & (pl.col("game_type") == "REG")).iter_rows(named=True):
        for t, o, ts, os_ in ((r["home_team"], r["away_team"], r["home_score"], r["away_score"]),
                              (r["away_team"], r["home_team"], r["away_score"], r["home_score"])):
            out.setdefault((r["season"], t), []).append((r["gameday"], "W" if ts > os_ else "L" if ts < os_ else "T"))
    return out


def record_before(rec: dict, season: int, team: str, day: str) -> str:
    res = [x for d, x in rec.get((season, team), []) if d < day]
    w, l, t = res.count("W"), res.count("L"), res.count("T")
    return f"{w}-{l}" + (f"-{t}" if t else "")


def league_table(proj: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Every team's ratings for this week: its own game's row, or for a team on bye, its
    nearest earlier row (else the next one). Offence and defence, with percentiles 0-100."""
    rows = proj.filter(pl.col("season") == season).with_columns((pl.col("week") - week).alias("d"))
    rows = rows.with_columns(pl.when(pl.col("d") <= 0).then(-pl.col("d") * 2).otherwise(pl.col("d") * 2 + 1).alias("rank_d"))
    cols = [c for _, o, d in PCT_ROWS for c in (o, d)]
    t = rows.sort("rank_d").group_by("team").first().select("team", *cols)
    pct = []
    for c in cols:
        x = -pl.col(c) if c in DEF_LOWER_BETTER else pl.col(c)
        pct.append(((x.rank("average") - 1) / (pl.len() - 1) * 100).round(0).cast(pl.Int32).alias(f"p_{c}"))
    return t.with_columns(pct)


def margin_bars(g: dict) -> str:
    items = sorted(((gm.GROUPS[f], g[f"m_{f}"]) for f in gm.FEATS), key=lambda x: -abs(x[1]))
    items = [(lab, v) for lab, v in items if abs(v) >= 0.05]
    top = max([abs(v) for _, v in items] + [0.5])
    rows = []
    for lab, v in items:
        w = abs(v) / top * 100
        side = "home" if v > 0 else "away"
        rows.append(f'<div class="bar-row"><span class="bar-lab">{e(lab)}</span>'
                    f'<span class="bar-track"><span class="bar-half left">{"" if v > 0 else f"<i class=bar-fill style=width:{w:.1f}% data-side=away></i>"}</span>'
                    f'<span class="bar-half right">{f"<i class=bar-fill style=width:{w:.1f}% data-side=home></i>" if v > 0 else ""}</span></span>'
                    f'<span class="bar-val {side}">{abs(v):.1f}</span></div>')
    total = g["model_margin"]
    to = NAME[g["team"]] if total > 0 else NAME[g["a_team"]]
    return (f'<div class="bars">{"".join(rows)}</div>'
            f'<div class="bar-axis"><span>← {e(NICK[g["a_team"]])}</span><span>{e(NICK[g["team"]])} →</span></div>'
            f'<p class="note">Each bar is one input\'s contribution to the projected margin, in points, '
            f'drawn toward the team it favours. Together they come to {abs(total):.1f} toward {e(to)}.</p>')


def game_section(g: dict, lt: pl.DataFrame, rec: dict) -> str:
    a, h = g["a_team"], g["team"]
    day = dt.date.fromisoformat(g["gameday"])
    when = f'{day.strftime("%A, %B")} {day.day}' + (f' · {clock(g["gametime"])}' if g["gametime"] else "")
    line, model = g["spread_line"], g["model_margin"]
    played = g["points"] is not None
    if line is not None:
        gap = model - line
        side = h if gap > 0 else a
        side_line = -line if side == h else line
        pick = f'{NAME[side]} {side_line:+.1f}'.replace("+0.0", "PK")
        number = (f'<p class="lede">The model makes this <b>{e(fav(g, model))}</b>. The market has it '
                  f'<b>{e(fav(g, line))}</b>, a gap of <b>{abs(gap):.1f}</b> points toward {e(NAME[side])}.</p>')
        diff = f"{abs(gap):.1f}"
    else:
        gap, pick, diff = None, "no line yet", "–"
        number = f'<p class="lede">The model makes this <b>{e(fav(g, model))}</b>. No market line yet.</p>'
    t = total_info(g)
    if t["line"] is not None:
        number += (f'<p class="lede">Total: the model has <b>{t["mt"]:.1f}</b> points, the market <b>{t["line"]:.1f}</b>'
                   + (f', so the model side is the <b>{t["side"].lower()}</b> by {abs(t["gap"]):.1f}.</p>' if t["side"] else ".</p>"))
    result = ""
    if played:
        res = g["points"] - g["a_points"]
        final = f'Final: {NICK[a]} {g["a_points"]}, {NICK[h]} {g["points"]}.'
        if gap is not None and abs(gap) > 0:
            ats = (res - line) * (1 if gap > 0 else -1)
            verdict = "covered" if ats > 0 else "lost" if ats < 0 else "pushed"
            chip = "win" if ats > 0 else "loss" if ats < 0 else "push"
            result = f'<p class="final"><span class="chip {chip}">{verdict}</span> {e(final)} The model side, {e(pick)}, {verdict}.</p>'
        else:
            result = f'<p class="final">{e(final)}</p>'
        if t["res"] is not None:
            result += f'<p class="final">{globals()["chip"](t["res"], True)} The {t["side"].lower()} {t["line"]:.1f} {"won" if t["res"] > 0 else "lost" if t["res"] < 0 else "pushed"} ({g["points"] + g["a_points"]} points).</p>'
    qb = "".join(f'<tr><td>{e(n or "TBD")}</td><td>{e(NAME[t])}</td><td class="num">{v:+.3f}</td></tr>'
                 for n, t, v in ((g["a_qb_name"], a, g["a_qb_adj"]), (g["qb_name"], h, g["qb_adj"])))
    ta, th = lt.filter(pl.col("team") == a).row(0, named=True), lt.filter(pl.col("team") == h).row(0, named=True)
    pct = "".join(f'<tr><th scope="row">{lab}</th>' + "".join(
        f'<td><span class="pct" style="--p:{t[f"p_{c}"]}">{t[f"p_{c}"]}</span></td>'
        for t, c in ((ta, o), (th, o), (ta, d), (th, d))) + "</tr>" for lab, o, d in PCT_ROWS)
    league = {c: lt[c].mean() for c in lt.columns if c != "team" and not c.startswith("p_")}
    rate = "".join(f'<tr><th scope="row">{lab}</th>' + "".join(
        f'<td class="num">{fmt.format(t[f"{s}_{k}"] - league[f"{s}_{k}"])}</td>'
        for s, t in (("off", ta), ("off", th), ("def", ta), ("def", th))) + "</tr>" for lab, k, fmt in RATING_ROWS)
    big = gap is not None and abs(gap) >= BIG_GAP
    return f'''
<article class="game" id="g-{g["game_id"]}">
  <header class="game-head">
    <p class="eyebrow">{e(when)} · {e(g["stadium"] or "")}</p>
    <div class="score">
      <div class="side"><span class="team">{e(NAME[a])}</span><span class="rec">{record_before(rec, g["season"], a, g["gameday"])}</span><span class="pts" data-side="away">{g["a_proj"]:.1f}</span></div>
      <span class="at">at</span>
      <div class="side"><span class="team">{e(NAME[h])}</span><span class="rec">{record_before(rec, g["season"], h, g["gameday"])}</span><span class="pts" data-side="home">{g["proj"]:.1f}</span></div>
    </div>
    {result}
  </header>
  <section><h3>The number{' <span class="chip flag">gap 3+</span>' if big else ''}</h3>{number}
    <dl class="kv"><div><dt>Vegas line</dt><dd>{e(fav(g, line) if line is not None else "no line yet")}</dd></div>
    <div><dt>Model</dt><dd>{e(fav(g, model))}</dd></div><div><dt>Difference</dt><dd>{diff}</dd></div>
    <div><dt>Model side</dt><dd>{e(pick)}</dd></div>
    <div><dt>Vegas total</dt><dd>{"–" if t["line"] is None else f'{t["line"]:.1f}'}</dd></div>
    <div><dt>Model total</dt><dd>{t["mt"]:.1f}</dd></div>
    <div><dt>Total side</dt><dd>{e(f'{t["side"]} {t["line"]:.1f}' if t["side"] else "–")}</dd></div></dl></section>
  <section><h3>Where the margin comes from</h3>{margin_bars(g)}</section>
  <section><h3>Quarterbacks</h3>
    <div class="scroll"><table><thead><tr><th>Listed starter</th><th>Team</th><th class="num">Adjustment, EPA / dropback</th></tr></thead><tbody>{qb}</tbody></table></div>
    <p class="note">The listed starter's rating against the passers who actually played in the games behind the team ratings. Zero means he is the quarterback already in the numbers.</p></section>
  <section><h3>How they rate <span class="sub">percentile vs the league, 100 = best (pass rate and pace: 100 = most)</span></h3>
    <div class="scroll"><table class="pct-table"><thead><tr><th></th><th>{e(NICK[a])} off</th><th>{e(NICK[h])} off</th><th>{e(NICK[a])} def</th><th>{e(NICK[h])} def</th></tr></thead><tbody>{pct}</tbody></table></div></section>
  <section><h3>Team ratings <span class="sub">per play, vs league average</span></h3>
    <div class="scroll"><table><thead><tr><th></th><th class="num">{e(NICK[a])} off</th><th class="num">{e(NICK[h])} off</th><th class="num">{e(NICK[a])} def</th><th class="num">{e(NICK[h])} def</th></tr></thead><tbody>{rate}</tbody></table></div>
    <p class="note">Opponent adjusted, with last season fading in over the first weeks. On defence, negative is good. Rest: {e(NAME[a])} {g["a_rest_days"]} days, {e(NAME[h])} {g["rest_days"]} days.</p></section>
  <p class="back"><a href="#top">Back to the board</a></p>
</article>'''


def total_info(g: dict) -> dict:
    """Model total vs the Vegas total: side, gap and (for played games) the graded result."""
    mt, line = g["proj"] + g["a_proj"], g["total_line"]
    out = {"mt": mt, "line": line, "side": None, "gap": None, "res": None}
    if line is not None:
        out["gap"] = mt - line
        out["side"] = "Over" if mt > line else "Under" if mt < line else None
        if g["points"] is not None and out["side"]:
            out["res"] = (g["points"] + g["a_points"] - line) * (1 if out["side"] == "Over" else -1)
    return out


def chip(res: float, long: bool = False) -> str:
    cls = "win" if res > 0 else "loss" if res < 0 else "push"
    txt = ("won" if res > 0 else "lost" if res < 0 else "pushed") if long else ("W" if res > 0 else "L" if res < 0 else "P")
    return f'<span class="chip {cls}">{txt}</span>'


def board_card(g: dict, rec: dict) -> str:
    a, h, line, model = g["a_team"], g["team"], g["spread_line"], g["model_margin"]
    gap = None if line is None else model - line
    side = None if gap is None else (h if gap > 0 else a)
    pick = "–" if side is None else f'{NICK[side]} {(-line if side == h else line):+.1f} ({abs(gap):.1f})'
    status = ""
    t = total_info(g)
    tot = "–" if t["line"] is None else f'{t["mt"]:.1f} vs {t["line"]:.1f}' + (f' · {t["side"]}' if t["side"] else "")
    if g["points"] is not None:
        if gap is not None and abs(gap) > 0:
            status = "ATS " + chip((g["points"] - g["a_points"] - line) * (1 if gap > 0 else -1))
        if t["res"] is not None:
            status += " O/U " + chip(t["res"])
        status += f' <span class="fin">{g["a_points"]}-{g["points"]}</span>'
    flag = ' data-big="1"' if gap is not None and abs(gap) >= BIG_GAP else ""
    day = dt.date.fromisoformat(g["gameday"])
    return f'''<a class="card" href="#g-{g["game_id"]}"{flag}>
  <span class="when">{day.strftime("%a %b")} {day.day} · {e(clock(g["gametime"]))}</span>
  <span class="m"><span class="t">{e(NICK[a])} <small>{record_before(rec, g["season"], a, g["gameday"])}</small></span><span class="p" data-side="away">{g["a_proj"]:.1f}</span></span>
  <span class="m"><span class="t">{e(NICK[h])} <small>{record_before(rec, g["season"], h, g["gameday"])}</small></span><span class="p" data-side="home">{g["proj"]:.1f}</span></span>
  <span class="kv2"><span>Vegas</span><b>{e(fav(g, line) if line is not None else "–")}</b></span>
  <span class="kv2"><span>Model</span><b>{e(fav(g, model))}</b></span>
  <span class="kv2"><span>Side</span><b>{e(pick)}</b></span>
  <span class="kv2"><span>Total</span><b>{e(tot)}</b></span>
  <span class="status">{status}</span>
</a>'''


def render(season: int, weeks: list[int]) -> str:
    df = gm.build()
    proj = gm.project(df)
    g = gm.games(proj)
    sched = pl.read_parquet(gm.DATA / "raw" / "schedules.parquet").with_columns(pl.col("home_team", "away_team").replace(gm.RENAMES))
    rec = records(sched)
    back = gm.record(g.filter(pl.col("season").is_in([2022, 2023, 2024, 2025])))
    back3 = gm.record(g.filter(pl.col("season").is_in([2022, 2023, 2024, 2025])), BIG_GAP)
    cur = gm.record(g.filter(pl.col("season") == season))
    cur3 = gm.record(g.filter(pl.col("season") == season), BIG_GAP)
    tcur = gm.total_record(g.filter(pl.col("season") == season))
    tback = gm.total_record(g.filter(pl.col("season").is_in([2022, 2023, 2024, 2025])))
    weeks_html, games_html = [], []
    for wk in weeks:
        wg = g.filter((pl.col("season") == season) & (pl.col("week") == wk)).sort("gameday", "gametime")
        lt = league_table(proj, season, wk)
        rows = list(wg.iter_rows(named=True))
        n_big = sum(1 for r in rows if r["spread_line"] is not None and abs(r["model_margin"] - r["spread_line"]) >= BIG_GAP)
        weeks_html.append(f'<section class="week"><h2>Week {wk}</h2><p class="wk-meta">{len(rows)} games · {n_big} with the model 3+ points off the market</p>'
                          f'<div class="board">{"".join(board_card(r, rec) for r in rows)}</div></section>')
        games_html += [game_section(r, lt, rec) for r in rows]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f'''<title>NFL Game Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap" id="top">
  <header class="masthead">
    <p class="eyebrow">{season} season · model updated {stamp}</p>
    <h1>NFL Game Board</h1>
    <p class="intro">A projected score for every game, built from play-by-play efficiency adjusted for opponent, beside the market spread and total. Open any game for where the margin comes from.</p>
    <div class="tally">
      <div><span class="k">{season} against the spread</span><span class="v">{cur["w"]}-{cur["l"]}-{cur["p"]}</span><span class="s">{cur["pct"]:.1%} · {cur3["w"]}-{cur3["l"]} on 3+ gaps</span></div>
      <div><span class="k">2022-2025 backtest</span><span class="v">{back["w"]}-{back["l"]}-{back["p"]}</span><span class="s">{back["pct"]:.1%} · {back3["w"]}-{back3["l"]} on 3+ gaps</span></div>
      <div><span class="k">{season} over/under</span><span class="v">{tcur["w"]}-{tcur["l"]}-{tcur["p"]}</span><span class="s">{tcur["pct"]:.1%} · 2022-2025 {tback["w"]}-{tback["l"]} ({tback["pct"]:.1%})</span></div>
      <div><span class="k">Margin error, 2022-2025</span><span class="v">{back["mae_model"]:.2f}</span><span class="s">market {back["mae_line"]:.2f} pts per game</span></div>
    </div>
    <p class="caveat">Graded against closing lines, the model tracks the market rather than beating it: 52.4% is breakeven at -110, and its misses run larger than the market's, on spreads and on totals. Treat a big gap as a question to look into, not a bet.</p>
  </header>
  {"".join(weeks_html)}
  <div class="games">{"".join(games_html)}</div>
  <footer><p>Model output, not betting advice. Every number on this page comes from <code>scripts/game_model.py</code>. The only hand input is the list of players ruled out ahead of the injury report (<code>props_log/manual_out.csv</code>).</p></footer>
</div>'''


CSS = '''
/* Layout: one column; a board of game cards, then a breakdown per game. Away = left/amber, home = right/blue. */
:root{
  --bg:#f3f5f4; --surface:#ffffff; --ink:#15201b; --muted:#5b6b64; --line:#d5ddd9;
  --away:#b4651c; --home:#2d5c96; --accent:#1d6b45; --win:#1d7a46; --loss:#b23b32; --flag:#8a5a00;
  --f-display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;
  --f-body:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-num:"JetBrains Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1513; --surface:#17201c; --ink:#e3ebe7; --muted:#97a8a0; --line:#2b3833;
  --away:#e09a55; --home:#79a6e0; --accent:#5cc28d; --win:#5cc28d; --loss:#f07e74; --flag:#e6b450; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f1513; --surface:#17201c; --ink:#e3ebe7; --muted:#97a8a0; --line:#2b3833;
  --away:#e09a55; --home:#79a6e0; --accent:#5cc28d; --win:#5cc28d; --loss:#f07e74; --flag:#e6b450; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.5 var(--f-body)}
.wrap{max-width:68rem;margin:0 auto;padding-inline:16px;padding-block:24px 48px;display:grid;gap:40px}
h1,h2,h3{font-family:var(--f-display);font-weight:700;letter-spacing:.01em;text-wrap:balance;margin:0}
h1{font-size:clamp(2.4rem,6vw,3.6rem);line-height:1;text-transform:uppercase}
h2{font-size:1.8rem;text-transform:uppercase}
h3{font-size:1.15rem;text-transform:uppercase;letter-spacing:.06em;display:flex;flex-wrap:wrap;align-items:baseline;gap:.5rem}
.sub{font-family:var(--f-body);font-size:.8rem;font-weight:400;letter-spacing:0;text-transform:none;color:var(--muted)}
.eyebrow,.wk-meta,.when{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
.intro{max-width:62ch;margin:.5rem 0 0;color:var(--muted)}
.masthead{display:grid;gap:12px}
.tally{display:grid;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-top:8px}
.tally>div{background:var(--surface);padding:14px 16px;display:grid;gap:2px}
.tally .k{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.tally .v{font:600 1.7rem/1.1 var(--f-num);font-variant-numeric:tabular-nums}
.tally .s{font-size:.85rem;color:var(--muted)}
.caveat{max-width:70ch;font-size:.9rem;color:var(--muted);margin:0}
.week{display:grid;gap:12px}
.board{display:grid;grid-template-columns:repeat(auto-fill,minmax(15rem,1fr));gap:12px}
.card{display:grid;gap:4px;padding:14px 16px;background:var(--surface);border:1px solid var(--line);color:inherit;text-decoration:none}
.card[data-big]{border-color:var(--flag)}
.card:hover,.card:focus-visible{border-color:var(--accent);outline:none}
.card .m{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--f-display);font-size:1.35rem;font-weight:600}
.card small{font-family:var(--f-body);font-size:.75rem;color:var(--muted);font-weight:400}
.p,.pts{font-family:var(--f-num);font-variant-numeric:tabular-nums}
[data-side=away]{color:var(--away)} [data-side=home]{color:var(--home)}
.kv2{display:flex;justify-content:space-between;gap:8px;font-size:.88rem;border-top:1px dashed var(--line);padding-top:4px}
.kv2 span{color:var(--muted)} .kv2 b{font-weight:600;text-align:right}
.status{min-height:1.2rem;font:.85rem var(--f-num)}
.chip{display:inline-block;font:600 .7rem/1 var(--f-body);text-transform:uppercase;letter-spacing:.08em;padding:4px 7px;border:1px solid currentColor}
.chip.win{color:var(--win)} .chip.loss{color:var(--loss)} .chip.push{color:var(--muted)} .chip.flag{color:var(--flag)}
.games{display:grid;gap:56px}
.game{display:grid;gap:28px;border-top:3px solid var(--ink);padding-top:20px;scroll-margin-top:16px}
.game section{display:grid;gap:10px;min-width:0}
.score{display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:16px;margin-top:8px}
.side{display:grid;gap:2px;min-width:0}
.side:last-child{text-align:right}
.team{font-family:var(--f-display);font-size:clamp(1.3rem,3.5vw,2rem);font-weight:600;line-height:1.1}
.rec{font-size:.8rem;color:var(--muted)}
.pts{font-size:clamp(2.4rem,7vw,3.6rem);font-weight:600;line-height:1}
.at{font-family:var(--f-display);color:var(--muted);text-transform:uppercase;padding-bottom:.6rem}
.final{margin:8px 0 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.lede{margin:0;max-width:65ch;font-size:1.05rem}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:0}
.kv div{background:var(--surface);padding:10px 12px} .kv dt{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.kv dd{margin:2px 0 0;font:600 1rem var(--f-num)}
.bars{display:grid;gap:6px}
.bar-row{display:grid;grid-template-columns:minmax(7rem,9rem) 1fr 2.5rem;align-items:center;gap:10px;font-size:.9rem}
.bar-track{display:grid;grid-template-columns:1fr 1fr;height:14px;border-inline:0}
.bar-half{display:flex;height:100%;background:color-mix(in srgb,var(--line) 45%,transparent)}
.bar-half.left{justify-content:flex-end;border-right:2px solid var(--ink)}
.bar-fill{display:block;height:100%}
.bar-fill[data-side=away]{background:var(--away)} .bar-fill[data-side=home]{background:var(--home)}
.bar-val{font:600 .85rem var(--f-num);text-align:right} .bar-val.away{color:var(--away)} .bar-val.home{color:var(--home)}
.bar-axis{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);padding-inline:calc(min(9rem,30%) + 10px) calc(2.5rem + 10px)}
.note{margin:0;font-size:.85rem;color:var(--muted);max-width:70ch}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:30rem;font-size:.92rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600}
.num{text-align:right;font-family:var(--f-num);font-variant-numeric:tabular-nums}
.pct{display:inline-block;min-width:2.6rem;text-align:right;font:600 .9rem var(--f-num);padding:2px 6px;
  background:color-mix(in srgb,var(--accent) calc(var(--p) * 0.5%),transparent)}
.back a,footer a{color:var(--accent)}
footer{font-size:.85rem;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}
code{font-family:var(--f-num);font-size:.85em}
@media (max-width:520px){.bar-row{grid-template-columns:6.5rem 1fr 2.2rem}.bar-axis{padding-inline:calc(6.5rem + 10px) calc(2.2rem + 10px)}}
@media (prefers-reduced-motion:no-preference){.card{transition:border-color .15s}}
'''


def current_week(season: int) -> int:
    """The first week with a game today or later (scores can lag a day behind)."""
    today = dt.date.today().isoformat()
    s = pl.read_parquet(gm.DATA / "raw" / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG") & (pl.col("gameday") >= today))
    return s["week"].min()


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    season = args[0] if args else 2026
    weeks = args[1:] or [current_week(season) - 1, current_week(season)]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(season, weeks))
    print(f"wrote {OUT.relative_to(ROOT)} (season {season}, weeks {weeks})")
