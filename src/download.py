"""Download Championship (E1), League One (E2) and League Two (E3) results from football-data.co.uk into data/raw/."""
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
URL = "https://www.football-data.co.uk/mmz4281/{s}/{d}.csv"
DIVS = ["E1", "E2", "E3"]  # neighbouring divisions link promoted/relegated teams' ratings

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for d in DIVS:
        for s in SEASONS:
            r = requests.get(URL.format(s=s, d=d), timeout=60)
            r.raise_for_status()
            (RAW / f"{d}_{s}.csv").write_bytes(r.content)
            print("saved", d, s, len(r.content), "bytes")

if __name__ == "__main__":
    main()
