# 環境建置（conda env = smartinvest）
(windows11)

```bash
# 建立環境
conda create -n smartinvest python=3.11 -y
conda activate smartinvest

# 安裝常用科學套件
conda install pandas numpy pyarrow -y
pip install streamlit altair tqdm yfinance==0.2.37

TA-Lib 0.5.1 安裝指令（Windows 11 + Anaconda）
# 更新 conda 後，直接從 conda-forge 取 0.5.1（含 DLL）
conda update -n base -c defaults conda -y
conda config --add channels conda-forge
conda config --set channel_priority flexible
conda install ta-lib -y         # 0.5.1

pip／wheel 備援方案
pip install --upgrade pip setuptools wheel
# 0.4.28 Windows wheel 下載：
pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl

驗證
python -c "import talib, numpy as np, sys, pathlib; print('Python =>', pathlib.Path(sys.executable)); print('TA-Lib =>', talib.__version__); print('SMA demo =>', talib.SMA(np.arange(1,11),5))"
期望輸出（版本 0.5.1）：
TA-Lib => 0.5.1
SMA demo => [nan nan nan nan  3.  4.  5.  6.  7.  8.]

指令一覽
# M1 ─ 下載台股 OHLCV
python src/utils/download_tw.py

# M2 ─ 技術指標（SMA/EMA/RSI/MACD/ATR…）
python src/utils/make_features.py

# M3 ─ 批量回測（輸出 *_bt.csv＋rank.csv）
python src/batch_bt.py

# M4 ─ 啟動前端
streamlit run app.py
