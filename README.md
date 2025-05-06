# SmartInvest MVP(TW Stock)

## Features
- 批量回測排行
- Sidebar 即時篩選
- 資金曲線（倍率／金額）
- 一鍵下載 CSV

## Quick Start
``bash
conda create -n smartinvest python=3.11 -y
conda activate smartinvest
pip install -r requirements.txt
streamlit run app.py

``bash
# 0. clone 專案
git clone https://github.com/zzzHou01/smart_invest_mvp.git
cd smart_invest_mvp
``
# 1. 建 conda 環境
conda create -n smartinvest python=3.11 -y
conda activate smartinvest
``
# 2. 安裝相依
pip install -r requirements.txt          # 先快速跑起來
# TA-Lib 若安裝失敗 → 詳見 docs/SETUP.md
``
# 3. 產生資料與回測（首次執行）
python src/utils/download_tw.py          # M1：下載 raw parquet
python src/utils/make_features.py        # M2：技術指標
python src/batch_bt.py                   # M3：批量回測 ➜ reports/
``
# 4. 啟動前端
streamlit run app.py
``

## Project Structure
```text
smart_invest_mvp/
│  app.py              # Streamlit interface
├─data/
│   ├─raw/             # 原始 OHLCV parquet
│   └─feat/            # 加 TA-Lib 指標
├─reports/
│   ├─rank.csv
│   └─*_bt.csv         # 含 date 欄
└─src/
    ├─backtest.py      # 單檔回測
    ├─batch_bt.py      # 批量回測
    └─utils/           # 下載、特徵工程輔助腳本
``` 
