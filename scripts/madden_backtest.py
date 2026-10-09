"""Backtest: do Madden team unit ratings add anything to the market or the game model? (see README)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import polars as pl, numpy as np
from sklearn.linear_model import Ridge
U = ["qb", "ol", "wr", "te", "rb", "dl", "lb", "db"]
t = pl.read_parquet(str(Path(__file__).resolve().parent.parent / "data") + "/madden_team_ratings.parquet")
g = pl.read_parquet(str(Path(__file__).resolve().parent.parent / "data") + "/game_projections.parquet").filter(pl.col("season").is_in([2025, 2026]) & pl.col("points").is_not_null())
h = t.select("season", "week", pl.col("team"), *[pl.col(f"mad_{u}").alias(f"mh_{u}") for u in U])
a = t.select("season", "week", pl.col("team").alias("a_team"), *[pl.col(f"mad_{u}").alias(f"ma_{u}") for u in U])
g = g.join(h, on=["season", "week", "team"]).join(a, on=["season", "week", "a_team"])
g = g.with_columns((pl.col("points") - pl.col("a_points")).alias("res"), (pl.col("points") + pl.col("a_points")).alias("tot"))
# composite: offense units vs opposing defense units, home minus away
off = lambda s: (pl.col(f"m{s}_qb") * 3 + pl.col(f"m{s}_ol") + pl.col(f"m{s}_wr") + pl.col(f"m{s}_te") * 0.5 + pl.col(f"m{s}_rb") * 0.5) / 6
dfn = lambda s: (pl.col(f"m{s}_dl") + pl.col(f"m{s}_lb") + pl.col(f"m{s}_db")) / 3
g = g.with_columns((off("h") - off("a")).alias("d_off"), (dfn("h") - dfn("a")).alias("d_def"),
                   (off("h") + off("a") - dfn("h") - dfn("a")).alias("s_net"),
                   *[(pl.col(f"mh_{u}") - pl.col(f"ma_{u}")).alias(f"d_{u}") for u in U])
print("games:", g.filter(pl.col("season") == 2025).height, "(2025)", g.filter(pl.col("season") == 2026).height, "(2026)")
def corr(x, y, f):
    d = g.filter(f).drop_nulls([x, y]); return np.corrcoef(d[x].to_numpy(), d[y].to_numpy())[0, 1]
g = g.with_columns((pl.col("res") - pl.col("spread_line")).alias("vs_line"), (pl.col("res") - pl.col("model_margin")).alias("vs_model"),
                   (pl.col("tot") - pl.col("total_line")).alias("vs_total"), (pl.col("d_off") + pl.col("d_def")).alias("d_team"))
for lab, f in (("2025", pl.col("season") == 2025), ("2026", pl.col("season") == 2026)):
    print(f"{lab}: corr(madden team diff, margin) {corr('d_team', 'res', f):+.3f} | with spread {corr('d_team', 'spread_line', f):+.3f} | "
          f"with market miss (result-spread) {corr('d_team', 'vs_line', f):+.3f} | with our model's miss {corr('d_team', 'vs_model', f):+.3f} | "
          f"net offense vs market total miss {corr('s_net', 'vs_total', f):+.3f}")
# cross-validated (by week blocks, 2025) and 2025 -> 2026
def cv(target, cols, base, alpha=50.0):
    d = g.filter(pl.col("season") == 2025).drop_nulls(cols + [target]); wk = d["week"].to_numpy(); pred = np.zeros(d.height)
    for k in range(5):
        te = (wk % 5) == k; X = d.select(cols).to_numpy(); mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
        m = Ridge(alpha=alpha).fit((X[~te] - mu) / sd, d[target].to_numpy()[~te]); pred[te] = m.predict((X[te] - mu) / sd)
    X = d.select(cols).to_numpy(); mu, sd = X.mean(0), X.std(0) + 1e-9; m = Ridge(alpha=alpha).fit((X - mu) / sd, d[target].to_numpy())
    d6 = g.filter(pl.col("season") == 2026).drop_nulls(cols + [target]); p6 = m.predict((d6.select(cols).to_numpy() - mu) / sd)
    out = []
    for dd, p in ((d, pred), (d6, p6)):
        y = dd[target].to_numpy()
        mae0 = np.abs(y).mean(); mae1 = np.abs(y - p).mean()
        side = np.sign(p); hit = np.sign(y) * side; w, l = (hit > 0).sum(), (hit < 0).sum()
        out.append(f"MAE {mae0:.3f}->{mae1:.3f}, side {w}-{l} ({w/max(w+l,1):.1%})")
    return " | 2026: ".join(out)
UN = [f"d_{u}" for u in U]
print("\nfix the market's spread miss (result - spread):   2025 CV:", cv("vs_line", ["d_off", "d_def"], None))
print("   by unit:                                        2025 CV:", cv("vs_line", UN, None))
print("fix our game model's margin miss (result - model): 2025 CV:", cv("vs_model", ["d_off", "d_def"], None))
print("   by unit:                                        2025 CV:", cv("vs_model", UN, None))
print("fix the market's total miss (total - line):        2025 CV:", cv("vs_total", ["s_net"], None))

print("\nby part of season (corr of Madden diff with our model's miss / with market miss):")
for lab, f in (("2025 wk1-6", (pl.col("season") == 2025) & (pl.col("week") <= 6)), ("2025 wk7-12", (pl.col("season") == 2025) & pl.col("week").is_between(7, 12)),
               ("2025 wk13+", (pl.col("season") == 2025) & (pl.col("week") >= 13)), ("2026 wk1-4", pl.col("season") == 2026)):
    print(f"  {lab:12s} n={g.filter(f).height:3d}  model miss {corr('d_team', 'vs_model', f):+.3f}  market miss {corr('d_team', 'vs_line', f):+.3f}")
# direct mix: margin = model + k * madden diff, k from 2025, judged on 2026 (and the reverse)
for k in (0.0, 0.1, 0.2, 0.3, 0.5):
    r = []
    for s in (2025, 2026):
        d = g.filter(pl.col("season") == s)
        p = d["model_margin"].to_numpy() + k * d["d_team"].to_numpy(); y = d["res"].to_numpy(); sp = d["spread_line"].to_numpy()
        hit = np.sign(y - sp) * np.sign(p - sp); w, l = (hit > 0).sum(), (hit < 0).sum()
        r.append(f"{s}: MAE {np.abs(y - p).mean():.3f} ATS {w}-{l} ({w/(w+l):.1%})")
    print(f"  model + {k:.1f} x Madden diff   " + " | ".join(r))
