# src/batch_bt.py
import pandas as pd, numpy as np, pathlib, warnings, math
from backtest import run_strategy, START, END

RAW  = pathlib.Path(__file__).parents[1] / "data" / "feat"
OUT  = pathlib.Path(__file__).parents[1] / "reports"
OUT.mkdir(exist_ok=True)

def perf(df: pd.DataFrame):
    eq = df["equity"]
    daily_ret = eq.pct_change().dropna()
    ann_ret = (eq.iat[-1] ** (252/len(eq)) - 1) * 100
    sharpe  = np.sqrt(252) * daily_ret.mean()/daily_ret.std() if daily_ret.std()>0 else np.nan

    # ── 新的交易分段邏輯 ──────────────────
    entries = df.index[df["signal"].diff() == 1].tolist()
    exits   = df.index[df["signal"].diff() == -1].tolist()

    # 若最後仍持倉，將最後一列當作 exit
    if len(exits) < len(entries):
        exits.append(df.index[-1])

    trades, wins = 0, 0
    for ent, ex in zip(entries, exits):
        seg = df.loc[ent:ex]
        trade_ret = (seg["ret"] + 1).prod() - 1
        trades += 1
        if trade_ret > 0:
            wins += 1
    winrate = wins / trades * 100 if trades else np.nan
    # ─────────────────────────────────────

    return ann_ret, sharpe, trades, winrate

def main():
    rows=[]
    for f in RAW.glob("*.parquet"):
        tk=f.stem
        df = run_strategy(pd.read_parquet(f).reset_index())
        df.to_csv(OUT/f"{tk}_bt.csv", index=False)
        ann, shp, n, wr = perf(df)
        rows.append((tk, ann, shp, n, wr))
    cols=["Ticker","AnnRet%","Sharpe","Trades","WinRate%"]
    rank = pd.DataFrame(rows, columns=cols).round(3).sort_values("Sharpe", ascending=False)
    rank.to_csv(OUT/"rank.csv", index=False)
    print("Top-5 by Sharpe")
    print(rank.head())

if __name__=="__main__":
    # silencing divide warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
