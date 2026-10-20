"""Combine seasons into one tidy match table."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in sorted(RAW.glob("E*_*.csv")):
        df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        odds = [c for c in ["AvgH", "AvgD", "AvgA", "BbAvH", "BbAvD", "BbAvA"] if c in df.columns]
        d = df[keep + odds].copy()
        if "AvgH" not in d and "BbAvH" in d:
            d = d.rename(columns={"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"})
        d["div"] = f.stem[:2]
        d["season"] = "20" + f.stem[-4:-2] + "-" + f.stem[-2:]
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
        frames.append(d[keep + ["AvgH", "AvgD", "AvgA", "div", "season"]])
    m = pd.concat(frames).sort_values("Date").reset_index(drop=True)
    m[["FTHG", "FTAG"]] = m[["FTHG", "FTAG"]].astype(int)
    m.to_csv(OUT / "matches.csv", index=False)
    print(len(m), "matches", m.season.unique())

if __name__ == "__main__":
    main()
