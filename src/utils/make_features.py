# src/utils/make_features.py
"""
為每檔 parquet (raw OHLCV) 計算技術指標並輸出到 data/feat/
依賴：
    pip install ta-lib pandas pyarrow tqdm
"""
from pathlib import Path
import pandas as pd
import talib
from tqdm import tqdm

RAW_DIR   = Path(__file__).parents[2] / "data" / "raw"
FEAT_DIR  = Path(__file__).parents[2] / "data" / "feat"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# TA-Lib 函式 (可再擴充)
INDICATORS = {
    "SMA_20"  : lambda c: talib.SMA(c, timeperiod=20),
    "EMA_20"  : lambda c: talib.EMA(c, timeperiod=20),
    "RSI_14"  : lambda c: talib.RSI(c, timeperiod=14),
    "MACD"    : lambda c: talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0],
    "ATR_14"  : lambda h,l,c: talib.ATR(h, l, c, timeperiod=14),
}

def make_feat(file_path: Path):
    df = pd.read_parquet(file_path)
    df = df.sort_values("Date").reset_index(drop=True)

    # 基於收盤價的指標
    close = df["Close"].astype(float).values
    high  = df["High"].astype(float).values
    low   = df["Low"].astype(float).values

    df_feat = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # 單列指標
    df_feat["SMA_20"] = INDICATORS["SMA_20"](close)
    df_feat["EMA_20"] = INDICATORS["EMA_20"](close)
    df_feat["RSI_14"] = INDICATORS["RSI_14"](close)
    df_feat["MACD"]   = INDICATORS["MACD"](close)
    # 多列輸入指標
    df_feat["ATR_14"] = INDICATORS["ATR_14"](high, low, close)

    df_feat.to_parquet(FEAT_DIR / file_path.name, index=False)

def main():
    files = list(RAW_DIR.glob("*.parquet"))
    for f in tqdm(files, desc="Features"):
        make_feat(f)

if __name__ == "__main__":
    main()
