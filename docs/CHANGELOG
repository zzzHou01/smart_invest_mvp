# Change Log

## 2025-05-07
# 方案A yifanance
# 1. 套件是否安裝？
python -c "import yfinance, sys; print(yfinance.__version__)"
# → 只要能印出版本號，例如 0.2.37，就確定安裝 OK

# 2. 網路是否被擋？
python -c "import urllib.request, ssl, pprint; 
           url='https://query1.finance.yahoo.com/v7/finance/quote?symbols=AAPL';
           try:
               print(urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=5).read()[:100])
           except Exception as e:
               print('Error:', e)"
# 如果連這行都回傳空白或超時，就 100% 是網路/防火牆問題


# 方案B 改用FinMind
# 裝最新穩定版
pip install FinMind==1.7.8

#   ─ 或 ─

# 直接安裝最新（目前同樣會抓 1.7.8）
pip install FinMind

# 確認版本
python -c "import FinMind, sys; print(FinMind.__version__)"

--------------------------------------------------------------
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

# 執行.py
# python src/utils/download_tw.py

# 應看到 20 個 .parquet 檔（~250–350 KB/檔）代表 M1 資料階段完成。
# dir data\raw


# 使用TA-Lib	(version:0.5.1)
# 產生 20 檔技術指標 parquet
# 仍在 (smartinvest) 資料夾根目錄
# 在utils 資料夾裡面新增make_features.py
-------------------------------------------------------------------------
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
----------------------------------------------------------------------------------
python src/utils/make_features.py

# 檢查檔案
dir data\feat

# 列出產出的 parquet 檔 (確認 20 個)
Get-ChildItem -Path .\data\feat | Select-Object Name, Length, LastWriteTime

# 隨機挑一檔，看前 5 列是否包含技術指標欄位 (SMA_20、EMA_20、RSI_14…)
python -c "import pathlib, random, pandas as pd; f=random.choice(list(pathlib.Path('data/feat').glob('*.parquet'))); print('Sample file =>', f); print(pd.read_parquet(f).head())"

# 在 src\ 目錄新增 backtest.py，貼入下面程式碼並存檔：
--------------------------------------------------------------------
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
--------------------------------------------------------------------------------------------
# 執行單檔回測
# 最熱門的權值股 2330
python src/backtest.py           # 預設 ticker="2330"
# 成功時終端機會顯示
Backtest done → ...\reports\2330_bt.csv

# 驗收輸出
# 文件大小 + 最後 5 列，確認 equity curve 已計算
Get-Item .\reports\2330_bt.csv
python -c "import pandas as pd; df=pd.read_csv('reports/2330_bt.csv'); print(df.tail())"

建檔：src/batch_bt.py
--------------------------------------------------------------------------------------------
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
    trades  = (df["signal"].diff()==1).sum()
    wins    = ((df["ret"]>0)&(df["pos"].shift()==1)).sum()
    winrate = wins / trades * 100 if trades else np.nan
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
----------------------------------------------------------------------------------------------------
# 執行
python src/batch_bt.py

# 驗收
# 查看排行榜
type reports\rank.csv | select -First 7
------------------------------------------------------------------------------
# PowerShell 驗證指令
# 確認現在的工作目錄就是 smart_invest_mvp
(Get-Item .).FullName

# 列出 rank.csv
Get-ChildItem .\reports\rank.csv

# 若還是不行，用絕對路徑先跑通
RANK_FILE = r"C:\Users\jbb86\smart_invest_mvp\reports\rank.csv"
# 請改成電腦環境設定的絕對路徑
----------------------------------------------------------------------------------
# 建檔:在smart_invest_mvp(專案檔案根目錄)底下建立app.py
------------------
# app-A.py
# Step A ─── 純路徑測試，確保 rank.csv 讀得到
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent          # 專案根目錄
DATA_PATH = ROOT / "reports" / "rank.csv"       # rank.csv 路徑

df = pd.read_csv(DATA_PATH, encoding="utf-8")   # 若亂碼→改 'big5'
print(f" Read OK — shape={df.shape}")
print(df.head())
-------------------------------------------------------------------------------
cd C:\Users\jbb86\smart_invest_mvp
conda activate smartinvest
python app.py          # ← 先確認能印出前 5 列
# 只要看到 Read OK 與表格，就代表路徑與編碼都沒問題。

# Streamlit 版（Step B：真正的 MVP-0）
# 完成 Step A 後，把 app.py 換成下列內容（或覆蓋舊檔）：
----------------------------------------------------------------------------------
# app.py ─── SmartInvest Streamlit MVP-0
import streamlit as st
import pandas as pd
from pathlib import Path

# ── 路徑設定 ─────────────────────────
ROOT = Path(__file__).resolve().parent          # 根目錄
DATA_PATH = ROOT / "reports" / "rank.csv"       # 報表

# ── 頁面基本設定 ─────────────────────
st.set_page_config(page_title="SmartInvest – 回測排行榜",
                   layout="wide")

# ── 讀取資料 (使用快取避免重複 IO) ─────
@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")    # 亂碼可改 big5
    # 欄位順序 / 命名統一
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

df = load_rank(DATA_PATH)

# ── 畫面 ─────────────────────────────
st.title(" 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=600)                   # 核心表格
--------------------------------------------------------------------
cd C:\Users\jbb86\smart_invest_mvp
conda activate smartinvest
# 若沒裝 Streamlit：
pip install streamlit pandas

streamlit run app.py	# 若防火牆跳出「允許 Python 網路連線」對話框，點"允許"才能在瀏覽器連
---------------------------
# 注意事項：
# 結束 Streamlit 伺服器 → 回到 Terminal 按 Ctrl + C。
# 之後每次修改 app.py，瀏覽器會自動熱重載；若沒更新再按一次 R 或手動重新整理。
-------------------------------
# 做 S2 ：Sidebar 篩選器

| 篩選器            | 預設值       | 備註                             |
| -------------- | --------- | ------------------------------ |
| **Sharpe ≥**   | 0.0 → 2.0 | 用 `slider`，小數間隔 0.1            |
| **WinRate% ≥** | 0 → 700   | 你的 WinRate% 最大 600 多，故上限先給 700 |

# 修改 app.py（標示 ▲ 新增 部分）
------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

# ---------- 讀資料 ----------
df_full = load_rank(DATA_PATH)

# ---------- ▲ Sidebar 篩選 ----------
st.sidebar.header("篩選條件")

sharpe_min = st.sidebar.slider(
    "Sharpe ≥", 0.0, 2.0, 0.0, 0.1
)

winrate_min = st.sidebar.slider(
    "WinRate% ≥", 0.0, 700.0, 0.0, 10.0
)

# 套用條件
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

# ---------- 主畫面 ----------
st.title(" 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=600)
-----------------------------------------------------------------------------
# app.py 改完存檔後，Streamlit 會自動熱重載
# 若沒刷新就在瀏覽器按 Ctrl+R

# S3 - 圖表視覺化：分成 2 個小步驟
# S3-A：Top-N 年化報酬長條圖
# 修改/覆蓋 app.py內容:
---------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

# ... 前半段程式碼保持不變 ----------------------------
df_full = load_rank(DATA_PATH)

# -------- Sidebar 篩選 --------
st.sidebar.header("篩選條件")
sharpe_min = st.sidebar.slider("Sharpe ≥", 0.0, 2.0, 0.0, 0.1)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 700.0, 0.0, 10.0)
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

# ======== ▲ Top-N 長條圖控制 ========
st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N by AnnRet%", 1, 20, 10)
# ===================================

# -------- 主畫面 --------
st.title(" 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=400)  # 表格高度稍降，留空間放圖

# ======== ▲ 畫長條圖 ========
if not df.empty:
    # 先依 AnnRet% 由高到低排序再取前 N
    chart_df = (
        df.sort_values("AnnRet%", ascending=False)
          .head(int(top_n))            # Top N
          .set_index("Ticker")         # 把股票代號當 X 軸
    )
    st.subheader(f"Top {int(top_n)} AnnRet%")
    st.bar_chart(chart_df["AnnRet%"])
else:
    st.info("尚無符合條件的股票，調整篩選試試")
# =================================
---------------------------------------------------------------------------
# S3-B　資金曲線折線圖（每檔 _bt.csv）

目標：在長條圖下方，再顯示 1 檔股票的 Equity Curve
做法：
1.Sidebar 新增 Ticker 選擇器
2.讀取 reports\{Ticker}_bt.csv → 繪折線圖
3.若檔案不存在或欄位不符，顯示友善提示

# 先確定 _bt.csv 欄位
# 打開任何一檔，例如 reports\2330_bt.csv，通常會看到類似：
index,	Open,	High,	Low,		Close,	Volume,	SMA_20,	EMA_20,	RSI_14,	MACD,	ATR_14,	signal	,	pos,		ret,		equity
# (只要有 date（或 Date） 與 equity（或 Equity） 就能畫線)
# 修改 app.py
----------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

# ... 先前程式碼保持原樣 ------------------------------
df_full = load_rank(DATA_PATH)

# ---------- Sidebar 篩選 ----------
st.sidebar.header("篩選條件")
sharpe_min = st.sidebar.slider("Sharpe ≥", 0.0, 2.0, 0.0, 0.1)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 700.0, 0.0, 10.0)
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N by AnnRet%", 1, 20, 10)

# ======== ▲ 新增：Ticker 選擇器 =========
st.sidebar.markdown("---")
default_ticker = df["Ticker"].iloc[0] if not df.empty else df_full["Ticker"].iloc[0]
ticker_selected = st.sidebar.selectbox(
    "查看資金曲線（Ticker）",
    options=sorted(df_full["Ticker"].unique()),
    index=list(sorted(df_full["Ticker"].unique())).index(default_ticker)
)
# ======================================

# -------- 主畫面：表格 + Top-N Bar --------
st.title(" 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=350)

# --- Top-N bar chart (與之前相同) ---
if not df.empty:
    chart_df = (
        df.sort_values("AnnRet%", ascending=False)
          .head(int(top_n))
          .set_index("Ticker")
    )
    st.subheader(f"Top {int(top_n)} AnnRet%")
    st.bar_chart(chart_df["AnnRet%"])
else:
    st.info("尚無符合條件的股票，調整篩選試試")

# ======== ▲ 折線圖區塊 ==================
bt_path = ROOT / "reports" / f"{ticker_selected}_bt.csv"
st.markdown("---")
st.subheader(f"資金曲線 – {ticker_selected}")

@st.cache_data
def load_bt(path: Path):
    if path.exists():
        bt = pd.read_csv(path, parse_dates=["date"])
        # 若欄位名不同，請改成你的欄位
        bt = bt.rename(columns={"equity": "Equity"})
        return bt[["date", "Equity"]]
    return None

bt_df = load_bt(bt_path)
if bt_df is not None:
    # Altair 折線圖
    import altair as alt
    line = alt.Chart(bt_df).mark_line().encode(
        x="date:T",
        y="Equity:Q"
    ).properties(height=300)
    st.altair_chart(line, use_container_width=True)
else:
    st.warning(f"找不到 {bt_path.name}，或檔案格式不符")
# =======================================
--------------------------------------------------------------------------------------
# 出現問題
pd.read_csv(path, parse_dates=["date"]) 指定要把 date 欄位轉成時間格式，
但_bt.csv 首欄並不叫 date，而是 index（或可能沒有欄位名）。
因此 pandas 在驗證「date 欄位是否存在」時噴錯：
ValueError: Missing column provided to 'parse_dates': 'date'

# 嘗試解決方法:
# 先確認 _bt.csv 現況
# 檔頭長這樣：index,Open,High,Low,Close,Volume,SMA_20,EMA_20,RSI_14,MACD,ATR_14,signal,pos,ret,equity
# 沒有 date 欄，第一欄叫 index，且內容是 0, 1, 2 …（代表 bar 序號）。但是有 equity 欄（小寫）。
# 因此只要把 index 當作 X 軸（整數刻度），就能畫資金曲線；不用硬轉成日期。
# 修改 load_bt(), # 取代 Altair 畫圖段

# 修改app.py, 完整code如下：
-----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

# ... 先前程式碼保持原樣 ------------------------------
df_full = load_rank(DATA_PATH)

# ---------- Sidebar 篩選 ----------
st.sidebar.header("篩選條件")
sharpe_min = st.sidebar.slider("Sharpe ≥", 0.0, 2.0, 0.0, 0.1)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 700.0, 0.0, 10.0)
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N by AnnRet%", 1, 20, 10)

# ======== ▲ 新增：Ticker 選擇器 =========
st.sidebar.markdown("---")
default_ticker = df["Ticker"].iloc[0] if not df.empty else df_full["Ticker"].iloc[0]
ticker_selected = st.sidebar.selectbox(
    "查看資金曲線（Ticker）",
    options=sorted(df_full["Ticker"].unique()),
    index=list(sorted(df_full["Ticker"].unique())).index(default_ticker)
)
# ======================================

# -------- 主畫面：表格 + Top-N Bar --------
st.title(" 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=350)

# --- Top-N bar chart (與之前相同) ---
if not df.empty:
    chart_df = (
        df.sort_values("AnnRet%", ascending=False)
          .head(int(top_n))
          .set_index("Ticker")
    )
    st.subheader(f"Top {int(top_n)} AnnRet%")
    st.bar_chart(chart_df["AnnRet%"])
else:
    st.info("尚無符合條件的股票，調整篩選試試")

# ======== ▲ 折線圖區塊 ==================
bt_path = ROOT / "reports" / f"{ticker_selected}_bt.csv"
st.markdown("---")
st.subheader(f"資金曲線 – {ticker_selected}")

@st.cache_data
def load_bt(path: Path):
    if not path.exists():
        return None

    bt = pd.read_csv(path)

    # ── 1. 取得 X 軸欄 ─────────────────
    if "date" in bt.columns:
        x_col = "date"
        bt["date"] = pd.to_datetime(bt["date"])
    elif "Date" in bt.columns:
        x_col = "Date"
        bt["Date"] = pd.to_datetime(bt["Date"])
    else:
        # 沒有日期欄，就用 index 欄或 DataFrame 的 row index
        if "index" in bt.columns:
            x_col = "index"
        else:
            bt.reset_index(inplace=True)
            x_col = "index"

    # ── 2. Equity 欄統一大寫 ───────────
    if "equity" in bt.columns:
        bt = bt.rename(columns={"equity": "Equity"})
    elif "Equity" not in bt.columns:
        st.warning(f"{path.name} 缺少 equity 欄，無法畫圖")
        return None

    return bt[[x_col, "Equity"]].rename(columns={x_col: "X"})

bt_df = load_bt(bt_path)
if bt_df is not None:
    import altair as alt
    line = (
        alt.Chart(bt_df)
        .mark_line()
        .encode(x="X:Q", y="Equity:Q")
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)
else:
    st.warning(f"找不到 {bt_path.name}，或檔案格式不符")
-------------------------------------------------------------------------------------------------------------------
# S4 – 下載目前篩選結果（selected.csv）
# 功能目標

| 元件                   | 作用                                                        |
| -------------------- | --------------------------------------------------------- |
| `st.download_button` | 讓使用者把 **目前表格內容** 直接下載成 CSV                                |
| 檔名                   | `selected_{today}.csv`（自動附日期，例：`selected_2025-05-07.csv`） |
| 內容                   | `df` 的所有欄位（已經過篩選 & 排序）                                    |

# 修改 app.py（只新增 6 行，標 ▲ 新增）
# 把下面片段插在 表格與長條圖之間（可放在 st.dataframe() 之後）：
----------------------------------------------------------------------------------
# -------- 主畫面：表格 --------
st.title("📈 批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=350)

# ======== ▲ 下載選股按鈕 =========
from datetime import date
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="💾 下載目前篩選結果",
    data=csv_bytes,
    file_name=f"selected_{date.today()}.csv",
    mime="text/csv",
)
# =================================
----------------------------------------------------------------------------------
# to_csv(index=False)：不包含 DataFrame 索引
# utf-8-sig：開啟時 Excel 不會亂碼
# mime="text/csv"：讓瀏覽器正確提示下載

# 完成標準
# 按鈕可正常下載
# CSV 內容正確、無亂碼

# 完整app.py檔案如下：
---------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

# ... 先前程式碼保持原樣 ------------------------------
df_full = load_rank(DATA_PATH)

# ---------- Sidebar 篩選 ----------
st.sidebar.header("篩選條件")
sharpe_min = st.sidebar.slider("Sharpe ≥", 0.0, 2.0, 0.0, 0.1)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 700.0, 0.0, 10.0)
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N by AnnRet%", 1, 20, 10)

# ======== ▲ 新增：Ticker 選擇器 =========
st.sidebar.markdown("---")
default_ticker = df["Ticker"].iloc[0] if not df.empty else df_full["Ticker"].iloc[0]
ticker_selected = st.sidebar.selectbox(
    "查看資金曲線（Ticker）",
    options=sorted(df_full["Ticker"].unique()),
    index=list(sorted(df_full["Ticker"].unique())).index(default_ticker)
)
# ======================================

# -------- 主畫面：表格 --------
st.title("批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=350)

# ======== ▲ 下載選股按鈕 =========
from datetime import date
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="下載目前篩選結果",
    data=csv_bytes,
    file_name=f"selected_{date.today()}.csv",
    mime="text/csv",
)
# =================================

# --- Top-N bar chart (與之前相同) ---
if not df.empty:
    chart_df = (
        df.sort_values("AnnRet%", ascending=False)
          .head(int(top_n))
          .set_index("Ticker")
    )
    st.subheader(f"Top {int(top_n)} AnnRet%")
    st.bar_chart(chart_df["AnnRet%"])
else:
    st.info("尚無符合條件的股票，調整篩選試試")

# ======== ▲ 折線圖區塊 ==================
bt_path = ROOT / "reports" / f"{ticker_selected}_bt.csv"
st.markdown("---")
st.subheader(f"資金曲線 – {ticker_selected}")

@st.cache_data
def load_bt(path: Path):
    if not path.exists():
        return None

    bt = pd.read_csv(path)

    # ── 1. 取得 X 軸欄 ─────────────────
    if "date" in bt.columns:
        x_col = "date"
        bt["date"] = pd.to_datetime(bt["date"])
    elif "Date" in bt.columns:
        x_col = "Date"
        bt["Date"] = pd.to_datetime(bt["Date"])
    else:
        # 沒有日期欄，就用 index 欄或 DataFrame 的 row index
        if "index" in bt.columns:
            x_col = "index"
        else:
            bt.reset_index(inplace=True)
            x_col = "index"

    # ── 2. Equity 欄統一大寫 ───────────
    if "equity" in bt.columns:
        bt = bt.rename(columns={"equity": "Equity"})
    elif "Equity" not in bt.columns:
        st.warning(f"{path.name} 缺少 equity 欄，無法畫圖")
        return None

    return bt[[x_col, "Equity"]].rename(columns={x_col: "X"})

bt_df = load_bt(bt_path)
if bt_df is not None:
    import altair as alt
    line = (
        alt.Chart(bt_df)
        .mark_line()
        .encode(x="X:Q", y="Equity:Q")
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)
else:
    st.warning(f"找不到 {bt_path.name}，或檔案格式不符")
---------------------------------------------------------------------------------------------

# winRate%異常，數值應在0~100之間。
# 原因：src/batch_bt.py 裡 perf() 的「勝場計數」方式錯了：wins = ((df["ret"] > 0) & (df["pos"].shift() == 1)).sum()
# df["ret"] > 0 會把 持倉期間內的每一根 K 棒 都算成一次「勝」，
# 導致 wins 遠大於真正的交易筆數。
# 再除以 trades（完整開平倉次數），最後乘 100，當然變成 500、600…

# 排除方法:修改perf(), 完整code如下:（直接覆蓋 batch_bt.py）
-------------------------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------------------
------------------------------------------------------------------(debug)------------------------------------------
# 重新執行批量回測：
cd smart_invest_mvp/src
python batch_bt.py
# reports\rank.csv 會被覆寫, WinRate% 回到 0–100

# 在 app.py修改
# 在vs code的app.py檔案裏面按下ctrl+F鍵搜尋winrate_min並取代
# 取代程式碼如下(直接複製覆蓋掉舊的)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 100.0, 0.0, 1.0)

# 測試頁面展現
streamlit run app.py

# 更新後的完整 app.py（已修正「本金變動即時影響資金曲線」問題）。
# 

