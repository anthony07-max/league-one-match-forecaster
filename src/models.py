"""Elo and time-weighted Poisson (Dixon-Coles style, without the low-score correction) models."""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

HOME_ADV, K, BASE, NEW_TEAM = 60.0, 20.0, 1500.0, 1450.0

def elo_expect(rh, ra):
    return 1 / (1 + 10 ** ((ra - (rh + HOME_ADV)) / 400))

def run_elo(m: pd.DataFrame):
    """Walk forward through matches. Returns pre-match expectations and final ratings."""
    ratings, pre, history = {}, [], []
    for r in m.itertuples():
        rh = ratings.get(r.HomeTeam, NEW_TEAM if ratings else BASE)
        ra = ratings.get(r.AwayTeam, NEW_TEAM if ratings else BASE)
        e = elo_expect(rh, ra)
        s = 1.0 if r.FTHG > r.FTAG else 0.5 if r.FTHG == r.FTAG else 0.0
        gd = abs(r.FTHG - r.FTAG)
        mult = 1 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = K * mult * (s - e)
        ratings[r.HomeTeam], ratings[r.AwayTeam] = rh + delta, ra - delta
        pre.append(e)
        history.append((r.Date, r.HomeTeam, rh + delta)); history.append((r.Date, r.AwayTeam, ra - delta))
    hist = pd.DataFrame(history, columns=["Date", "team", "elo"])
    return np.array(pre), ratings, hist

def elo_probs(e, draw_base=0.27):
    """Convert Elo expected score to H/D/A, draw share shrinking as the match gets one-sided."""
    d = draw_base * (1 - abs(2 * e - 1)) + 0.05
    h = np.clip(e - d / 2, 0.01, 1)
    a = np.clip(1 - e - d / 2, 0.01, 1)
    tot = h + d + a
    return np.column_stack([h / tot, d / tot, a / tot])

class PoissonModel:
    def __init__(self, half_life_days=180):
        self.hl = half_life_days

    def fit(self, m: pd.DataFrame, as_of):
        m = m[m.Date < as_of]
        m = m[m.Date >= as_of - pd.Timedelta(days=730)]
        self.teams = sorted(set(m.HomeTeam) | set(m.AwayTeam))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        rows, y, w = [], [], []
        age = (as_of - m.Date).dt.days.values
        wt = 0.5 ** (age / self.hl)
        for (h, a, gh, ga), ww in zip(m[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].values, wt):
            x = np.zeros(2 * n + 1); x[idx[h]] = 1; x[n + idx[a]] = 1; x[-1] = 1
            rows.append(x); y.append(gh); w.append(ww)
            x = np.zeros(2 * n + 1); x[idx[a]] = 1; x[n + idx[h]] = 1
            rows.append(x); y.append(ga); w.append(ww)
        self.idx, self.n = idx, n
        self.reg = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(np.array(rows), y, sample_weight=w)
        return self

    def rates(self, h, a):
        c = self.reg.coef_; b = self.reg.intercept_
        att = lambda t: c[self.idx[t]] if t in self.idx else np.percentile(c[: self.n], 25)
        dfn = lambda t: c[self.n + self.idx[t]] if t in self.idx else np.percentile(c[self.n: 2 * self.n], 75)
        return np.exp(b + att(h) + dfn(a) + c[-1]), np.exp(b + att(a) + dfn(h))

    def probs(self, h, a, maxg=10):
        lh, la = self.rates(h, a)
        g = np.arange(maxg + 1)
        M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]), lh, la
