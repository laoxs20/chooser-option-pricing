import os
import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict

# 导入你的配置文件
from config import (
    START_DATE, END_DATE, RAW_DATA_DIR,
    FRED_API_KEY, ALPHA_VANTAGE_KEY,
    FRED_SERIES_ID, TICKER, VIX_TICKER
)


# ===================== 通用工具函数 =====================
def create_data_directory() -> None:
    """创建原始数据存储目录，不存在则自动生成"""
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    print(f"数据存储目录已确认: {RAW_DATA_DIR}")


def save_data(df: pd.DataFrame, file_name: str, save_parquet: bool = True, save_csv: bool = True) -> None:
    """
    标准化保存数据，默认同时保存parquet（高效压缩）和csv（兼容查看）
    :param df: 待保存的DataFrame
    :param file_name: 文件名（不含后缀）
    :param save_parquet: 是否保存parquet格式
    :param save_csv: 是否保存csv格式
    """
    create_data_directory()
    base_path = os.path.join(RAW_DATA_DIR, file_name)

    if save_parquet:
        parquet_path = f"{base_path}.parquet"
        df.to_parquet(parquet_path, index=True)
        print(f"数据已保存至: {parquet_path}")

    if save_csv:
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=True, encoding="utf-8-sig")
        print(f"数据已保存至: {csv_path}")


def validate_date_range(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    校验并过滤数据日期范围，严格匹配START_DATE和END_DATE
    :param df: 原始DataFrame
    :param date_col: 日期列名，若索引为日期则传"index"
    :return: 过滤后的DataFrame
    """
    if date_col == "index":
        df = df.loc[START_DATE:END_DATE].copy()
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[(df[date_col] >= START_DATE) & (df[date_col] <= END_DATE)].copy()
        df.set_index(date_col, inplace=True)

    # 按日期升序排序
    df.sort_index(inplace=True)
    print(f"数据日期范围校验完成: 共{len(df)}条交易日数据，起始{df.index.min().date()}，结束{df.index.max().date()}")
    return df


# ===================== 1. 雅虎财经数据拉取函数（核心免费数据源） =====================
def fetch_jpm_yahoo_finance() -> Optional[pd.DataFrame]:
    """
    从雅虎财经拉取JPM摩根大通2018-2024年日线股票数据
    包含：开盘价、最高价、最低价、收盘价、调整收盘价、成交量
    :return: 处理后的JPM日线DataFrame，失败返回None
    """
    print("\n==================== 开始拉取雅虎财经JPM股票数据 ====================")
    try:
        # 拉取日线数据
        ticker = yf.Ticker(TICKER)
        df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")

        # 数据清洗与标准化
        df.index = pd.to_datetime(df.index).tz_localize(None)  # 去除时区，统一格式
        df = validate_date_range(df, date_col="index")

        if df.empty:
            print("错误：雅虎财经JPM数据拉取结果为空")
            return None

        # 保存数据
        save_data(df, f"jpm_daily_yahoo_{START_DATE[:4]}_{END_DATE[:4]}")
        print("雅虎财经JPM股票数据拉取完成")
        return df

    except Exception as e:
        print(f"雅虎财经JPM数据拉取失败: {str(e)}")
        return None


def fetch_vix_yahoo_finance() -> Optional[pd.DataFrame]:
    """
    从雅虎财经拉取VIX恐慌指数2018-2024年日线数据
    :return: 处理后的VIX日线DataFrame，失败返回None
    """
    print("\n==================== 开始拉取雅虎财经VIX指数数据 ====================")
    try:
        ticker = yf.Ticker(VIX_TICKER)
        df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")

        # 数据清洗与标准化
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = validate_date_range(df, date_col="index")

        if df.empty:
            print("错误：雅虎财经VIX数据拉取结果为空")
            return None

        # 保存数据
        save_data(df, f"vix_daily_yahoo_{START_DATE[:4]}_{END_DATE[:4]}")
        print("雅虎财经VIX指数数据拉取完成")
        return df

    except Exception as e:
        print(f"雅虎财经VIX数据拉取失败: {str(e)}")
        return None


# ===================== 2. FRED美联储数据拉取函数（国债利率） =====================
def fetch_treasury_fred(series_id: str = FRED_SERIES_ID) -> Optional[pd.DataFrame]:
    """
    从FRED美联储数据库拉取国债收益率数据（默认3个月国债DGS3MO）
    :param series_id: FRED数据系列ID，默认使用config中的配置
    :return: 处理后的国债利率DataFrame，失败返回None
    """
    print(f"\n==================== 开始拉取FRED国债数据 {series_id} ====================")
    # 校验API Key
    if not FRED_API_KEY:
        print("错误：未配置FRED_API_KEY，请在.env文件中设置后重试")
        return None

    try:
        # FRED API请求配置
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": START_DATE,
            "observation_end": END_DATE,
            "frequency": "d",  # 日频数据
            "aggregation_method": "average"
        }

        # 发送请求
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 解析数据
        df = pd.DataFrame(data["observations"])
        df = df[["date", "value"]].copy()

        # 数据清洗
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # 处理缺失值
        df = validate_date_range(df, date_col="date")
        df.rename(columns={"value": series_id}, inplace=True)

        if df.empty:
            print(f"错误：FRED {series_id} 数据拉取结果为空")
            return None

        # 保存数据
        save_data(df, f"treasury_{series_id}_fred_{START_DATE[:4]}_{END_DATE[:4]}")
        print(f"FRED {series_id} 国债数据拉取完成")
        return df

    except Exception as e:
        print(f"FRED国债数据拉取失败: {str(e)}")
        return None


# ===================== 3. Alpha Vantage数据拉取函数（备用数据源） =====================
def fetch_jpm_alpha_vantage(outputsize: str = "full") -> Optional[pd.DataFrame]:
    """
    从Alpha Vantage拉取JPM摩根大通2018-2024年日线股票数据（备用数据源）
    :param outputsize: 数据量，full=全量历史数据，compact=最近100条
    :return: 处理后的JPM日线DataFrame，失败返回None
    """
    print("\n==================== 开始拉取Alpha Vantage JPM股票数据 ====================")
    # 校验API Key
    if not ALPHA_VANTAGE_KEY:
        print("错误：未配置ALPHA_VANTAGE_KEY，请在.env文件中设置后重试")
        return None

    try:
        # Alpha Vantage API请求配置
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": TICKER,
            "outputsize": outputsize,
            "apikey": ALPHA_VANTAGE_KEY,
            "datatype": "json"
        }

        # 发送请求（免费版5次/分钟限制，加延时避免触发限流）
        time.sleep(12)
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 解析数据
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            print(f"Alpha Vantage API返回异常: {data.get('Note', data.get('Information', '未知错误'))}")
            return None

        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")

        # 数据清洗与标准化
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="coerce")
        # 重命名列名，去除数字前缀，统一格式
        df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            },
            inplace=True
        )
        df = validate_date_range(df, date_col="index")

        if df.empty:
            print("错误：Alpha Vantage JPM数据拉取结果为空")
            return None

        # 保存数据
        save_data(df, f"jpm_daily_alpha_vantage_{START_DATE[:4]}_{END_DATE[:4]}")
        print("Alpha Vantage JPM股票数据拉取完成")
        return df

    except Exception as e:
        print(f"Alpha Vantage JPM数据拉取失败: {str(e)}")
        return None


# ===================== 批量拉取主函数（一键执行第一周数据提取） =====================
def fetch_all_first_week_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    一键拉取第一周所需的全部原始数据
    :return: 所有拉取结果的字典，key为数据名称，value为对应DataFrame
    """
    print(f"===== 开始执行第一周数据批量拉取，日期范围：{START_DATE} 至 {END_DATE} =====")

    # 按顺序拉取所有数据
    result = {
        "jpm_yahoo": fetch_jpm_yahoo_finance(),
        "vix_yahoo": fetch_vix_yahoo_finance(),
        "treasury_fred": fetch_treasury_fred(),
        "jpm_alpha_vantage": fetch_jpm_alpha_vantage()
    }

    # 统计拉取结果
    success_count = sum(1 for v in result.values() if v is not None)
    total_count = len(result)
    print(f"\n===== 批量拉取完成：成功{success_count}/{total_count}个数据源 =====")

    return result


# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 一键拉取所有第一周所需数据
    fetch_all_first_week_data()

    # 也可单独调用某个函数，示例：
    # fetch_jpm_yahoo_finance()
    # fetch_treasury_fred()