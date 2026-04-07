# scripts/common/trading_calendar.py
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def generate_trading_calendar(start_date=None, end_date=None, save=True):
    """
    通过 yfinance 获取 JPM 的历史数据，提取交易日索引作为交易日历
    该方法自动剔除周末和美股节假日，简单可靠
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = config.END_DATE

    print(f"正在通过 yfinance 生成 NYSE 交易日历: {start_date} 至 {end_date}")

    # 下载 JPM 日线数据（会自动剔除休市日）
    jpm = yf.download("JPM", start=start_date, end=end_date, progress=False)

    if jpm.empty:
        raise ValueError("yfinance 未返回数据，请检查网络或日期范围")

    # 提取交易日索引，并去除时区信息
    trading_days = jpm.index.tz_localize(None)

    print(f"交易日数量: {len(trading_days)}")
    print(f"首个交易日: {trading_days[0].date()}, 末个交易日: {trading_days[-1].date()}")

    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        calendar_path = config.PROCESSED_DATA_DIR / "trading_calendar.csv"
        pd.DataFrame({'date': trading_days}).to_csv(calendar_path, index=False)
        print(f"交易日历已保存至: {calendar_path}")

    return trading_days


def load_trading_calendar():
    """加载已保存的交易日历 CSV"""
    calendar_path = config.PROCESSED_DATA_DIR / "trading_calendar.csv"
    if not calendar_path.exists():
        print("交易日历文件不存在，正在重新生成...")
        generate_trading_calendar()
    df = pd.read_csv(calendar_path, parse_dates=['date'])
    return pd.DatetimeIndex(df['date'])


if __name__ == "__main__":
    # 单独运行此脚本时，生成并保存交易日历
    generate_trading_calendar()