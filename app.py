import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
import altair as alt

# ────────────── 基本路徑 ──────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "rank.csv"

st.set_page_config(page_title="SmartInvest – 回測排行榜", layout="wide")

# ────────────── 讀排行榜 ──────────────
@st.cache_data
def load_rank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df[["Ticker", "AnnRet%", "Sharpe", "Trades", "WinRate%"]]

df_full = load_rank(DATA_PATH)

# ────────────── Sidebar 篩選 ───────────
st.sidebar.header("篩選條件")
sharpe_min  = st.sidebar.slider("Sharpe ≥",   0.0, 2.0,   0.0, 0.1)
winrate_min = st.sidebar.slider("WinRate% ≥", 0.0, 100.0, 0.0, 1.0)
df = df_full.query("Sharpe >= @sharpe_min and `WinRate%` >= @winrate_min")

st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N by AnnRet%", 1, 20, 10)

# ── 本金 & 顯示模式 ─────────────────────
st.sidebar.markdown("---")
initial_cap = st.sidebar.number_input(
    "本金 (TWD)", min_value=10_000, max_value=10_000_000,
    value=1_000_000, step=50_000
)
view_mode = st.sidebar.radio(
    "資金曲線顯示", ["倍率 (×)", "實際金額 (TWD)"], horizontal=True
)

# ── Ticker 選擇 ─────────────────────────
st.sidebar.markdown("---")
default_ticker = df["Ticker"].iloc[0] if not df.empty else df_full["Ticker"].iloc[0]
ticker_selected = st.sidebar.selectbox(
    "查看資金曲線（Ticker）",
    options=sorted(df_full["Ticker"].unique()),
    index=list(sorted(df_full["Ticker"].unique())).index(default_ticker)
)

# ────────────── 主表格 ────────────────
st.title("批量回測結果表")
st.caption(f"資料來源：{DATA_PATH}")
st.dataframe(df, height=350)

# 下載按鈕
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下載目前篩選結果", csv_bytes,
    file_name=f"selected_{date.today()}.csv",
    mime="text/csv"
)

# ────────────── Top-N 長條圖 ───────────
if not df.empty:
    chart_df = (
        df.sort_values("AnnRet%", ascending=False)
          .head(int(top_n)).set_index("Ticker")
    )
    st.subheader(f"Top {int(top_n)} AnnRet%")
    st.bar_chart(chart_df["AnnRet%"])
else:
    st.info("尚無符合條件的股票，請調整篩選。")

# ────────────── 資金曲線 ───────────────
bt_path = ROOT / "reports" / f"{ticker_selected}_bt.csv"
st.markdown("---")
st.subheader(f"資金曲線 – {ticker_selected}")

@st.cache_data
def _load_bt_raw(path: Path):
    """只讀取 X & Equity，其他換算放快取外層"""
    if not path.exists():
        return None
    bt = pd.read_csv(path)

    # 取得 X 軸
    if "date" in bt.columns:
        x_col = "date"; bt["date"] = pd.to_datetime(bt["date"])
    elif "Date" in bt.columns:
        x_col = "Date"; bt["Date"] = pd.to_datetime(bt["Date"])
    else:
        x_col = "index" if "index" in bt.columns else bt.reset_index().columns[0]

    # 統一 Equity 欄名
    if "equity" in bt.columns:
        bt = bt.rename(columns={"equity": "Equity"})
    elif "Equity" not in bt.columns:
        return None

    return bt[[x_col, "Equity"]].rename(columns={x_col: "X"})

bt_raw = _load_bt_raw(bt_path)

if bt_raw is not None:
    # ☆☆☆ 依使用者本金動態計算 Cash ☆☆☆
    bt_df = bt_raw.copy()
    bt_df["Cash"] = bt_df["Equity"] * initial_cap

    y_field = "Equity:Q" if view_mode.startswith("倍率") else "Cash:Q"
    line = (
        alt.Chart(bt_df)
        .mark_line()
        .encode(x="X:Q", y=y_field)
        .properties(height=300)
    )
    st.altair_chart(line, use_container_width=True)

    # 本金/期末資金/盈虧
    final_cash = bt_df["Cash"].iat[-1]
    abs_profit = final_cash - initial_cap
    st.caption(
        f"初始本金：NT$ {initial_cap:,.0f}　|　"
        f"期末資金：NT$ {final_cash:,.0f}　|　"
        f"盈虧：{'+' if abs_profit>=0 else ''}{abs_profit:,.0f}"
    )
else:
    st.warning(f"找不到 {bt_path.name}，或檔案格式不符")
