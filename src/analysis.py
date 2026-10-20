"""Backtest Elo vs Poisson vs bookmaker odds, rate teams, and simulate the current season."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import run_elo, elo_probs, PoissonModel

ROOT = Path(__file__).resolve().parents[1]
P, C = ROOT / "data" / "processed", ROOT / "charts"
C.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

def logloss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))

def brier(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))

def main():
    m = pd.read_csv(P / "matches.csv", parse_dates=["Date"])
    y_all = m.FTR.map({"H": 0, "D": 1, "A": 2}).values
    e, ratings, hist = run_elo(m)
    m["elo_e"] = e

    # --- Backtest on the full 2025-26 season, refitting Poisson weekly ---
    test_season = "2025-26"
    t = m[(m.season == test_season) & (m["div"] == "E2")].copy()
    yt = t.FTR.map({"H": 0, "D": 1, "A": 2}).values
    p_elo = elo_probs(t.elo_e.values)
    p_poi = []
    model, last_fit = None, None
    for r in t.itertuples():
        if last_fit is None or (r.Date - last_fit).days >= 7:
            model = PoissonModel().fit(m, r.Date); last_fit = r.Date
        p_poi.append(model.probs(r.HomeTeam, r.AwayTeam)[0])
    p_poi = np.array(p_poi)
    odds = 1 / t[["AvgH", "AvgD", "AvgA"]].values
    p_mkt = odds / odds.sum(1, keepdims=True)
    p_base = np.tile(np.bincount(y_all[(m.season != test_season).values & (m["div"] == "E2").values], minlength=3) / ((m.season != test_season) & (m["div"] == "E2")).sum(), (len(t), 1))
    bt = pd.DataFrame({
        "model": ["Base rates", "Elo", "Poisson", "Bookmaker average (implied)"],
        "log_loss": [logloss(p, yt) for p in [p_base, p_elo, p_poi, p_mkt]],
        "brier": [brier(p, yt) for p in [p_base, p_elo, p_poi, p_mkt]],
        "accuracy": [float((p.argmax(1) == yt).mean()) for p in [p_base, p_elo, p_poi, p_mkt]],
    })
    bt.to_csv(P / "backtest.csv", index=False)
    print(bt)

    # Calibration chart for Poisson home-win probabilities
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in [("Poisson", p_poi), ("Elo", p_elo), ("Bookmakers", p_mkt)]:
        b = np.digitize(p[:, 0], bins) - 1
        xs = [p[b == i, 0].mean() for i in range(10) if (b == i).sum() >= 15]
        ys = [(yt[b == i] == 0).mean() for i in range(10) if (b == i).sum() >= 15]
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Predicted home-win probability"); ax.set_ylabel("Observed home-win rate")
    ax.set_title(f"Calibration, League One {test_season}"); ax.legend(); fig.tight_layout(); fig.savefig(C / "02_calibration.png", dpi=150); plt.close()

    # --- Current season ---
    cur_season = m.season.max()
    cur = m[(m.season == cur_season) & (m["div"] == "E2")]
    teams = sorted(set(cur.HomeTeam) | set(cur.AwayTeam))
    as_of = cur.Date.max() + pd.Timedelta(days=1)
    model = PoissonModel().fit(m, as_of)
    c, n = model.reg.coef_, model.n
    rt = pd.DataFrame({"team": teams,
                       "elo": [ratings[t] for t in teams],
                       "attack": [np.exp(c[model.idx[t]]) for t in teams],
                       "defence_conceded": [np.exp(c[n + model.idx[t]]) for t in teams]}).sort_values("elo", ascending=False)
    rt.to_csv(P / "ratings.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(rt.team[::-1], rt.elo[::-1], color=["#c8102e" if t == "Leyton Orient" else "#7a8ca3" for t in rt.team[::-1]])
    ax.set_xlim(rt.elo.min() - 40, rt.elo.max() + 20); ax.set_title(f"League One Elo ratings, {as_of:%d %b %Y}")
    fig.tight_layout(); fig.savefig(C / "01_elo_ratings.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(rt.attack, rt.defence_conceded, color="#7a8ca3")
    for r in rt.itertuples():
        ax.annotate(r.team, (r.attack, r.defence_conceded), fontsize=7, color="#c8102e" if r.team == "Leyton Orient" else "k")
    ax.invert_yaxis(); ax.set_xlabel("Attack strength (higher = scores more)"); ax.set_ylabel("Goals-conceded factor (top = tighter defence)")
    ax.set_title("Poisson team strengths (time-weighted, last 2 years)"); fig.tight_layout(); fig.savefig(C / "03_attack_defence.png", dpi=150); plt.close()

    top = rt.team.head(6).tolist() + (["Leyton Orient"] if "Leyton Orient" in teams and "Leyton Orient" not in rt.team.head(6).tolist() else [])
    fig, ax = plt.subplots(figsize=(10, 5))
    for tm in top:
        h = hist[(hist.team == tm) & (hist.Date >= "2023-07-01")]
        ax.plot(h.Date, h.elo, label=tm, lw=2.5 if tm == "Leyton Orient" else 1.2)
    ax.legend(fontsize=8); ax.set_title("Elo history since 2023-24"); fig.tight_layout(); fig.savefig(C / "04_elo_history.png", dpi=150); plt.close()

    # Monte Carlo of remaining fixtures (each pair plays home and away once)
    played = set(zip(cur.HomeTeam, cur.AwayTeam))
    remaining = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    pts0 = {t: 0 for t in teams}; gd0 = {t: 0 for t in teams}
    for r in cur.itertuples():
        gd0[r.HomeTeam] += r.FTHG - r.FTAG; gd0[r.AwayTeam] += r.FTAG - r.FTHG
        if r.FTR == "H": pts0[r.HomeTeam] += 3
        elif r.FTR == "A": pts0[r.AwayTeam] += 3
        else: pts0[r.HomeTeam] += 1; pts0[r.AwayTeam] += 1
    lam = np.array([model.rates(h, a) for h, a in remaining])
    N = 10000
    gh = rng.poisson(lam[:, 0], size=(N, len(remaining))); ga = rng.poisson(lam[:, 1], size=(N, len(remaining)))
    ti = {t: i for i, t in enumerate(teams)}
    pts = np.tile([pts0[t] for t in teams], (N, 1)).astype(float)
    gd = np.tile([gd0[t] for t in teams], (N, 1)).astype(float)
    for j, (h, a) in enumerate(remaining):
        hw, dr = gh[:, j] > ga[:, j], gh[:, j] == ga[:, j]
        pts[:, ti[h]] += 3 * hw + dr; pts[:, ti[a]] += 3 * (~hw & ~dr) + dr
        gd[:, ti[h]] += gh[:, j] - ga[:, j]; gd[:, ti[a]] += ga[:, j] - gh[:, j]
    score = pts + gd / 1000 + rng.random(pts.shape) / 1e6
    pos = (-score).argsort(1).argsort(1) + 1
    sim = pd.DataFrame({"team": teams,
                        "points_now": [pts0[t] for t in teams],
                        "played": [((cur.HomeTeam == t) | (cur.AwayTeam == t)).sum() for t in teams],
                        "exp_points": pts.mean(0).round(1),
                        "p_title": (pos == 1).mean(0), "p_auto_promotion": (pos <= 2).mean(0),
                        "p_playoffs": ((pos >= 3) & (pos <= 6)).mean(0), "p_relegation": (pos >= len(teams) - 3).mean(0),
                        "exp_position": pos.mean(0).round(1)}).sort_values("exp_points", ascending=False)
    sim.to_csv(P / "season_simulation.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    s = sim[::-1]
    ax.barh(s.team, s.p_auto_promotion, color="#2a9d8f", label="Top 2")
    ax.barh(s.team, s.p_playoffs, left=s.p_auto_promotion, color="#e9c46a", label="Play-offs (3rd-6th)")
    ax.barh(s.team, s.p_relegation, left=1 - s.p_relegation, color="#e76f51", label="Relegation (bottom 4)")
    ax.set_xlim(0, 1); ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"Simulated {cur_season} outcomes ({N:,} runs, data to {cur.Date.max():%d %b %Y})")
    fig.tight_layout(); fig.savefig(C / "05_season_simulation.png", dpi=150); plt.close()

    # Upcoming probabilities for every remaining fixture (for the dashboard)
    rows = []
    for h, a in remaining:
        p, lh, la = model.probs(h, a)
        rows.append({"home": h, "away": a, "p_home": p[0], "p_draw": p[1], "p_away": p[2], "xg_home": lh, "xg_away": la})
    pd.DataFrame(rows).to_csv(P / "fixture_probs.csv", index=False)

    kf = {"data_to": f"{cur.Date.max():%Y-%m-%d}", "matches": int(len(m)), "test_season": test_season,
          "backtest": bt.round(4).to_dict("records"),
          "home_win_rate": float((m[m["div"]=="E2"].FTR == "H").mean()), "draw_rate": float((m[m["div"]=="E2"].FTR == "D").mean()),
          "top_elo": rt.head(5)[["team", "elo"]].round(0).to_dict("records"),
          "sim_top": sim.head(6)[["team", "points_now", "exp_points", "p_title", "p_auto_promotion", "p_playoffs"]].round(3).to_dict("records"),
          "orient": sim[sim.team == "Leyton Orient"].round(3).to_dict("records")}
    (P / "key_findings.json").write_text(json.dumps(kf, indent=2, default=str))
    print(json.dumps(kf, indent=2, default=str))

if __name__ == "__main__":
    main()
