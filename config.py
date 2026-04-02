# config.py
import os
from pathlib import Path

# 尝试加载 .env 文件（需要先安装 python-dotenv）
try:
    from dotenv import load_dotenv
    # 寻找项目根目录下的 .env 文件
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("where is env?")
    # 如果没有安装 dotenv，则忽略，直接使用系统环境变量
    pass

# ========== 时间范围 ==========
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# ========== 数据存储路径 ==========
DATA_DIR = "./data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"

# ========== API Keys（从环境变量读取） ==========
# 优先从环境变量获取，如果不存在则给出提示
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("警告：未找到 FRED_API_KEY 环境变量。请设置 FRED_API_KEY 或在 .env 文件中定义。")

# 可选：Alpha Vantage Key（后续可能用到）
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# ========== 数据标识 ==========
FRED_SERIES_ID = "DGS3MO"   # 3个月国债收益率
TICKER = "JPM"
VIX_TICKER = "^VIX"