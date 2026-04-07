# scripts/data_process/process_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_raw_data():
    """加载原始CSV，统一日期列为索引（列名均为小写 date）"""
    raw_dir = config.RAW_DATA_DIR

    # JPM
    jpm = pd.read_csv(raw_dir / "jpm_raw.csv", parse_dates=['date'])
    jpm = jpm.rename(columns={'date': 'Date'}).set_index('Date')
    # 只保留收盘价和成交量
    jpm = jpm[['Close', 'Volume']]

    # VIX (注意你的文件列名可能是：date, close, open, high, low, 涨跌幅)
    vix = pd.read_csv(raw_dir / "VIX_raw.csv", parse_dates=['date'])
    vix = vix.rename(columns={'date': 'Date', 'close': 'VIX'}).set_index('Date')
    vix = vix[['VIX']]  # 只保留收盘价作为VIX值

    # 利率 (列名 date, Treasury_Rate)
    rate = pd.read_csv(raw_dir / "DGS3MO_raw.csv", parse_dates=['date'])
    rate = rate.rename(columns={'date': 'Date', 'Treasury_Rate': 'Treasury_Rate'}).set_index('Date')
    rate = rate[['Treasury_Rate']]

    return jpm, vix, rate


def load_trading_calendar():
    """读取你已经生成的交易日历 (data/processed/trading_calendar.csv)"""
    cal_path = config.PROCESSED_DATA_DIR / "trading_calendar.csv"
    cal = pd.read_csv(cal_path, parse_dates=['Date'])
    return pd.DatetimeIndex(cal['Date'])


def clean_and_align(jpm, vix, rate, trading_days):
    """对齐到交易日历，处理缺失，生成基础列"""
    # 创建标准交易日DataFrame
    df = pd.DataFrame(index=trading_days)

    # 左连接
    df = df.join(jpm, how='left')
    df = df.join(vix, how='left')
    df = df.join(rate, how='left')

    # 统计缺失比例（用于报告）
    missing_stats = df.isnull().sum() / len(df) * 100
    print("缺失比例:\n", missing_stats)

    # 对VIX和利率进行前向填充（交易日历内但数据缺失）
    df[['VIX', 'Treasury_Rate']] = df[['VIX', 'Treasury_Rate']].ffill()

    # 如果JPM收盘价仍有缺失，用前一天填充
    df['Close'] = df['Close'].ffill()

    # 计算日收益率
    df['Return'] = df['Close'].pct_change()

    return df


def add_features(df):
    """构造至少10个特征（符合项目要求）"""
    # 1. 滚动波动率（20日，年化）
    df['Volatility_20d'] = df['Return'].rolling(20).std() * np.sqrt(252)

    # 2. 滚动波动率（60日）
    df['Volatility_60d'] = df['Return'].rolling(60).std() * np.sqrt(252)

    # 3. 成交量变化率（5日）
    df['Volume_Change_5d'] = df['Volume'].pct_change(5)

    # 4. VIX-JPM相关性（20日滚动）
    df['VIX_JPMA_Corr_20d'] = df['Return'].rolling(20).corr(df['VIX'].pct_change())

    # 5. 利率动量（5日变化）
    df['Rate_Momentum_5d'] = df['Treasury_Rate'].diff(5)

    # 6. 情绪得分（基于VIX变化率，归一化到0-1）
    vix_change = df['VIX'].pct_change(5)
    # 避免除零和全NaN
    if vix_change.max() != vix_change.min():
        df['Sentiment_Score'] = (vix_change - vix_change.min()) / (vix_change.max() - vix_change.min())
    else:
        df['Sentiment_Score'] = 0.5
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0.5)  # 缺失填充中性

    # 7. 股息增长率（论文参数2.33%，可改为常数）
    df['Dividend_Growth'] = 0.0233

    # 8. 滞后1期收益率
    df['Return_Lag1'] = df['Return'].shift(1)

    # 9. 波动率比率（20日/60日）
    df['Volatility_Ratio'] = df['Volatility_20d'] / df['Volatility_60d']

    # 10. 隐含波动率代理（直接用VIX百分比）
    df['Implied_Vol'] = df['VIX'] / 100.0

    # 可选：额外特征 11. 时间到决策日（固定0.5年）
    df['Time_To_Decision'] = 0.5

    return df


def main():
    print("=== 第二周：数据清洗与特征工程（CSV存储）===")

    # 1. 加载原始数据
    jpm, vix, rate = load_raw_data()

    # 2. 加载交易日历
    trading_days = load_trading_calendar()

    # 3. 清洗对齐
    df = clean_and_align(jpm, vix, rate, trading_days)

    # 4. 特征工程
    df = add_features(df)

    # 5. 确保只保留交易日
    df = df[df.index.isin(trading_days)]

    # 6. 保存为CSV（不再用parquet）
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_DIR / "market_data.csv"
    df.to_csv(out_path, encoding='utf-8-sig')
    print(f"清洗后数据已保存至 {out_path}")
    print(f"最终数据形状: {df.shape}")
    print(f"特征列表: {list(df.columns)}")

    # 可选：输出缺失统计报告
    missing_report = df.isnull().sum()
    print("\n缺失值统计（按列）:\n", missing_report[missing_report > 0])


if __name__ == "__main__":
    main()