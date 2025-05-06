# src/backtest.py
"""
讀取 data/feat/ 單檔資料，
以最簡單的「收盤 > SMA_20 進場、收盤 < SMA_20 出場」做多策略，
回測期間 2019-01-02 ~ 2024-12-31，
輸出 equity curve 與每筆交易摘要。
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parents[1] / "data" / "feat"
START, END = "2019-01-02", "2024-12-31"

def run_strategy(df: pd.DataFrame):
    df = df.set_index("Date").astype(float).loc[START:END].copy()
    df["signal"] = np.where(df["Close"] > df["SMA_20"], 1, 0)      # 持倉 = 1 / 0
    df["pos"]    = df["signal"].shift().fillna(0)                   # 當日開盤持倉
    df["ret"]    = df["Close"].pct_change().fillna(0)
    df["equity"] = (1 + df["ret"] * df["pos"]).cumprod()
    return df

def main(ticker="2330"):
    file = DATA_DIR / f"{ticker}.parquet"
    if not file.exists():
        raise FileNotFoundError(file)
    df_bt = run_strategy(pd.read_parquet(file).reset_index())
    out = Path(__file__).parents[1] / "reports"
    out.mkdir(exist_ok=True)
    df_bt.to_csv(out / f"{ticker}_bt.csv")
    print(f"Backtest done → {out / f'{ticker}_bt.csv'}")

if __name__ == "__main__":
    main()
