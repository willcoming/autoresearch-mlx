目標
最大化固定 validation folds 的 val_sharpe，
但不要明顯惡化 median_turnover、median_drawdown、median_exposure。

目前 baseline
- input recency bias = linear 0.95 -> 1.05
- mapping = sigmoid(logits - 2.5)
- val_sharpe = 2.248084
- median_turnover = 0.093135
- median_drawdown = 0.017512
- median_exposure = 0.102516
- fold_sharpes = [1.9402, 2.5559, 2.9957, -0.3006]

Immutable
- prepare.py
- data/GOOGL_1d_5y.csv
- fold definitions
- transaction cost model
- final test split

Mutable
- train.py only

每一輪規則
1. 先提出並執行一個最小實驗
2. 每輪只能做一個 contained change
3. 只能修改 train.py
4. 執行 `uv run train.py > run.log 2>&1`
5. 讀取：
   - val_sharpe
   - median_turnover
   - median_drawdown
   - median_exposure
   - fold_sharpes（若可取得）
6. 判斷 keep 或 revert
7. 若 keep，更新 baseline
8. 若 revert，還原到上一個 baseline
9. 將本輪結果寫入 results.tsv

決策原則
- 優先改善 weakest fold，尤其是 fold_4
- 不要只是靠壓低 exposure 來換 Sharpe
- 不要破壞目前低 turnover、低 drawdown 的 baseline 優勢
- 優先保留簡單、乾淨、可解釋的改動
- 不要再優先做 mapping offset / slope 搜尋
- 不要再優先做 recency weight 類微調
- 下一步優先考慮非常小的 temporal aggregation / temporal projection

實質改善定義
至少滿足以下之一，且其他核心指標未明顯惡化：
- val_sharpe 提升 >= 0.01
- weakest fold 有明顯改善
- median_turnover 明顯下降
- median_drawdown 明顯下降

Tie-breaker
若新版本與目前 baseline 的差異非常小，則保留：
- 較簡單的版本
- 參數較少的版本
- 較可解釋的版本
- 不需要新增結構的版本

立即停止條件
- 最多 5 輪
- 連續 2 輪沒有可接受改善
- 最近 3 輪內沒有實質改善
- 改善主要來自 exposure 明顯下降
- 新版本增加了額外參數、結構或複雜度，但只帶來極小幅度指標變動
- weakest fold 長期沒有被改善，且同類型微調已反覆證明效果有限
- 新實驗開始出現明顯指標交換，而不是真改善
- 需要修改 immutable 檔案才能繼續
- 無法提出合理的單一最小實驗

當觸發停止條件時
1. 停止提出新實驗
2. 輸出：「目前 baseline 視為暫時最佳解」
3. 總結：
   - 目前 baseline 的核心設計
   - 關鍵指標
   - weakest fold 狀態
   - 為什麼停止
   - 下一階段若要繼續，應換哪一類搜尋方向，而不是繼續微調同一類旋鈕