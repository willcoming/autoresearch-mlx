# BTC 日線回測

這個目錄是獨立的 BTC/USDT 日線回測版本，和 `btc/` 的 5 分 K 版本分開使用。

## 檔案

- `btc_prepare.py`: 下載 Binance BTC/USDT 日線資料並存到快取
- `btc_strategy.py`: 策略參數與進出場 signal 組合
- `btc_backtest.py`: 執行單次回測
- `btc_auto.py`: 自動隨機搜尋參數與 signal 組合
- `requirements.txt`: 依賴套件

## 需求

- Python 3.10 以上
- 可連線到 Binance API 的網路環境

## 安裝

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r btc1d/requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r btc1d\requirements.txt
```

## 第一次使用

先下載日線資料：

```bash
python3 btc1d/btc_prepare.py
```

下載完成後，快取會在 `btc1d/cache/btc_1d.csv`。

## 單次回測

```bash
python3 btc1d/btc_backtest.py
```

你會看到：

- sharpe
- profit_factor
- win_rate
- max_drawdown_pct
- total_return_pct
- num_trades

## 修改策略

編輯 `btc1d/btc_strategy.py`：

- 調整 RSI / ZLSMA / MACD 參數
- 修改 `ENTRY_SIGNALS`
- 修改 `EXIT_SIGNALS`

目前 signal 名稱如下：

- `macd_bull`
- `macd_bear`
- `rsi_exit_oversold`
- `rsi_exit_overbought`
- `price_breakout_zlsma`
- `price_breakdown_zlsma`
- `ema5_cross_over_10`
- `ema5_cross_under_10`
- `consecutive2_day_up`
- `consecutive2_day_down`

`ENTRY_SIGNALS` 和 `EXIT_SIGNALS` 都是 AND 邏輯，清單中的條件必須同時成立才會觸發。

## 自動優化

```bash
python3 btc1d/btc_auto.py
```

它會重複：

1. 隨機改一個參數或切換 signal 組合
2. 跑一次回測
3. 如果 sharpe 變好就保留，否則還原

輸出檔案：

- `btc1d/btc1d_results.tsv`
- `btc1d/btc1d_run.log`

按 `Ctrl+C` 可停止。

## 搬到另一台電腦

最少需要帶這個目錄：

- `btc1d/`

如果你不想重新下載資料，也可以一起帶：

- `btc1d/cache/btc_1d.csv`

如果另一台電腦無法連 Binance API，`btc_prepare.py` 會失敗。這種情況下可以：

- 直接把 `btc1d/cache/btc_1d.csv` 一起複製過去
- 或把 `btc_prepare.py` 裡的交易所改成你可用的來源
