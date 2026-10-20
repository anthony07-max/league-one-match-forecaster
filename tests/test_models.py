import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import elo_expect, elo_probs, PoissonModel

def test_elo_home_advantage():
    assert elo_expect(1500, 1500) > 0.5

def test_elo_probs_sum_to_one():
    p = elo_probs(np.array([0.2, 0.5, 0.8]))
    assert np.allclose(p.sum(1), 1)

def test_poisson_probs_valid():
    m = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=6, freq="7D"),
                      "HomeTeam": ["A", "B", "C", "A", "B", "C"], "AwayTeam": ["B", "C", "A", "C", "A", "B"],
                      "FTHG": [2, 1, 0, 3, 1, 1], "FTAG": [0, 1, 2, 1, 2, 0]})
    p, lh, la = PoissonModel().fit(m, pd.Timestamp("2025-03-01")).probs("A", "B")
    assert abs(p.sum() - 1) < 1e-3 and lh > 0 and la > 0
