# scripts/baseline_evaluation/evaluate_bsm.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.bsm_model.chooser_option import ChooserOptionBSM
import config

# ==================== 参数设置 ====================
SPREAD = 0.02  # 买卖价差 2%
VOL_WINDOW_BENCH = 252  # 生成基准时使用的波动率窗口（长期）
VOL_WINDOW_PRED = 20  # BSM 预测时使用的波动率窗口（短期）
T1 = 0.5
T2 = 1.0
K = 150
Q = 0.0233  # 股息率（论文固定值）

# ==================== 加载数据 ====================
df = pd.read_csv(config.PROCESSED_DATA_DIR / "market_data.csv", index_col=0, parse_dates=True)
# 确保日期排序
df = df.sort_index()
# 计算日收益率（用于滚动波动率）
df['Return'] = df['Close'].pct_change()

# ==================== 逐日计算基准价格和预测价格 ====================
bench_prices = []  # 基准价格（标签）
bsm_prices = []  # BSM 预测价格（模型输出）
dates = []

for i, (date, row) in enumerate(df.iterrows()):
    S0 = row['Close']
    r = row['Treasury_Rate'] / 100  # 注意：原始数据是百分比，转为小数

    # 跳过前 VOL_WINDOW_BENCH 天，以保证有足够历史计算波动率
    if i < max(VOL_WINDOW_BENCH, VOL_WINDOW_PRED):
        continue

    # 1. 计算基准价格使用的波动率（长期历史）
    hist_returns_bench = df['Return'].iloc[i - VOL_WINDOW_BENCH:i]
    sigma_bench = hist_returns_bench.std() * np.sqrt(252)

    # 2. 计算 BSM 预测使用的波动率（短期历史）
    hist_returns_pred = df['Return'].iloc[i - VOL_WINDOW_PRED:i]
    sigma_pred = hist_returns_pred.std() * np.sqrt(252)

    # 3. 实例化定价器（基准用大模拟次数，预测也用相同参数但不同 sigma）
    chooser_bench = ChooserOptionBSM(
        S0=S0, K=K, T1=T1, T2=T2, r=r, sigma=sigma_bench, q=Q, n_sim=50000, seed=42
    )
    chooser_pred = ChooserOptionBSM(
        S0=S0, K=K, T1=T1, T2=T2, r=r, sigma=sigma_pred, q=Q, n_sim=1, seed=42
    )

    # 基准价格 = 蒙特卡洛理论价格 × (1 + spread)
    price_bench_theory = chooser_bench.price_mc()
    price_bench = price_bench_theory * (1 + SPREAD)

    # BSM 预测价格（使用解析公式，速度快且无抽样误差）
    price_pred = chooser_pred.price_analytical()

    bench_prices.append(price_bench)
    bsm_prices.append(price_pred)
    dates.append(date)

    if len(dates) % 500 == 0:
        print(f"已处理 {len(dates)} 个交易日...")

# 转为 DataFrame
result_df = pd.DataFrame({
    'Date': dates,
    'Benchmark_Price': bench_prices,
    'BSM_Prediction': bsm_prices
})
result_df.set_index('Date', inplace=True)

# ==================== 计算误差 ====================
errors = result_df['BSM_Prediction'] - result_df['Benchmark_Price']
mae = errors.abs().mean()
rmse = np.sqrt((errors ** 2).mean())

print("\n========== BSM 模型性能评估（方案2：蒙特卡洛+价差） ==========")
print(f"MAE (平均绝对误差): {mae:.4f} 美元")
print(f"RMSE (均方根误差):  {rmse:.4f} 美元")
print(f"基准价格均值: {result_df['Benchmark_Price'].mean():.2f} 美元")
print(f"BSM预测均值:   {result_df['BSM_Prediction'].mean():.2f} 美元")

# ==================== 保存结果 ====================
result_df.to_csv(config.PROCESSED_DATA_DIR / "bsm_evaluation_results.csv")
print(f"\n结果已保存至 {config.PROCESSED_DATA_DIR / 'bsm_evaluation_results.csv'}")

# ==================== 可视化 ====================
plt.figure(figsize=(12, 8))

# 子图1：价格对比
plt.subplot(2, 1, 1)
plt.plot(result_df.index, result_df['Benchmark_Price'], label='Benchmark (MC+spread)', alpha=0.7)
plt.plot(result_df.index, result_df['BSM_Prediction'], label='BSM Prediction (short-term vol)', alpha=0.7)
plt.title('Chooser Option Price: Benchmark vs BSM Prediction')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# 子图2：误差
plt.subplot(2, 1, 2)
plt.fill_between(result_df.index, 0, errors, where=(errors > 0), color='red', alpha=0.3, label='Overpricing')
plt.fill_between(result_df.index, 0, errors, where=(errors < 0), color='green', alpha=0.3, label='Underpricing')
plt.plot(result_df.index, errors, color='black', linewidth=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Prediction Error (BSM - Benchmark)')
plt.ylabel('Error (USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(config.PROCESSED_DATA_DIR / "bsm_error_plot.png", dpi=150)
plt.show()

