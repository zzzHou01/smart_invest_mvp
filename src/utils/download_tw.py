# src/utils/download_tw.py
from FinMind.data import DataLoader
from pathlib import Path
from tqdm import tqdm

TICKERS = [
    "2330","2317","2303","2454","2412",
    "1301","2882","2881","2891","2603",
    "2308","1101","2002","3008","1216",
    "2886","2892","2207","2301","5871"
]
START, END = "2018-01-01", "2024-12-31"

def main():
    dl = DataLoader()          # 免費訪客 600 req/hr 已足夠；註冊 token 可升到 6000 req/hr
    out_dir = Path(__file__).parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tk in tqdm(TICKERS, desc="FinMind"):
        df = dl.taiwan_stock_daily(stock_id=tk, start_date=START, end_date=END)
        if df.empty:
            print(f"[WARN] {tk} 無資料，跳過"); continue
        df = df.rename(columns={
            "open":"Open","max":"High","min":"Low",
            "close":"Close","Trading_Volume":"Volume","date":"Date"})
        df["Adj Close"] = df["Close"]          # 與 yfinance 欄位一致
        df.to_parquet(out_dir / f"{tk}.parquet", index=False)

if __name__ == "__main__":
    main()
