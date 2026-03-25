"""
btc_strategy.py — 此檔案由自主迴圈修改
目前最佳策略參數與進出場邏輯。
對應 Pine Script: S01_strategy.pine（基準線）

可用的進場/出場信號（在下方 ENTRY_SIGNALS / EXIT_SIGNALS 中組合）：
  進場候選：
    macd_bull             MACD 柱狀圖動能向上（反轉或連續3根遞增）
    rsi_exit_oversold     RSI 從超賣區回升（穿越 RSI_OVERSOLD）
    price_breakout_zlsma  收盤價向上突破 ZLSMA
    ema5_cross_over_10    EMA5 上穿 EMA10
    consecutive2_day_up   連續 2 根陽線

  出場候選：
    macd_bear             MACD 柱狀圖動能向下
    rsi_exit_overbought   RSI 從超買區回落（穿越 RSI_OVERBOUGHT）
    price_breakdown_zlsma 收盤價向下跌破 ZLSMA
    ema5_cross_under_10   EMA5 下穿 EMA10
    consecutive2_day_down 連續 2 根陰線
"""

# === RSI 參數 ===
RSI_LENGTH    = 15
RSI_OVERSOLD  = 30
RSI_OVERBOUGHT = 70

# === ZLSMA 參數 ===
ZLSMA_LENGTH = 55

# === MACD 參數 ===
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# === 手續費（雙邊）===
COMMISSION = 0.001  # 0.1% 進場 + 0.1% 出場

# === 進出場邏輯（AND 組合，所有條件同時成立才觸發）===
ENTRY_SIGNALS = ["macd_bull", "rsi_exit_oversold", "price_breakout_zlsma"]
EXIT_SIGNALS  = ["macd_bear", "rsi_exit_overbought"]
