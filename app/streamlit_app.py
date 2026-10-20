import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

P = Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="League One forecaster", layout="wide")
kf = json.loads((P / "key_findings.json").read_text())
st.title("League One match forecaster")
st.caption(f"Elo + Poisson models on football-data.co.uk results, data to {kf['data_to']}. By Anthony Adusei-Poku (@anthony07-max)")

sim = pd.read_csv(P / "season_simulation.csv")
fx = pd.read_csv(P / "fixture_probs.csv")
rt = pd.read_csv(P / "ratings.csv")
bt = pd.read_csv(P / "backtest.csv")

t1, t2, t3, t4 = st.tabs(["Season simulation", "Match predictor", "Ratings", "Model backtest"])
with t1:
    st.dataframe(sim.style.format({c: "{:.1%}" for c in ["p_title", "p_auto_promotion", "p_playoffs", "p_relegation"]}), use_container_width=True, hide_index=True)
with t2:
    teams = sorted(rt.team)
    c1, c2 = st.columns(2)
    h = c1.selectbox("Home", teams, index=teams.index("Leyton Orient") if "Leyton Orient" in teams else 0)
    a = c2.selectbox("Away", [t for t in teams if t != h])
    row = fx[(fx.home == h) & (fx.away == a)]
    if row.empty:
        st.info("This fixture has already been played this season.")
    else:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{h} win", f"{r.p_home:.0%}"); m2.metric("Draw", f"{r.p_draw:.0%}"); m3.metric(f"{a} win", f"{r.p_away:.0%}")
        m4.metric("Expected goals", f"{r.xg_home:.2f} - {r.xg_away:.2f}")
with t3:
    st.plotly_chart(px.scatter(rt, x="attack", y="defence_conceded", text="team", color="elo", title="Attack vs goals conceded (lower = better defence)"), use_container_width=True)
    st.dataframe(rt, hide_index=True, use_container_width=True)
with t4:
    st.dataframe(bt, hide_index=True)
    st.image(str(P.parents[1] / "charts" / "02_calibration.png"))
