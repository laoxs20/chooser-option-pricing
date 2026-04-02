import config
from data_fetcher import fetch_stock_data, fetch_vix_data, fetch_fred_rate, save_raw_data, fetch_yfinance_data, \
    fetch_jpm_akshare


def main():
    # 1. JPM 股价
    try:
        jpm_df = fetch_jpm_akshare(config.START_DATE, config.END_DATE)
        if not jpm_df.empty:
            save_raw_data(jpm_df, f"{config.TICKER}_raw.csv")
        else:
            print("Warning: JPM data is empty.")
    except Exception as e:
        print(f"Failed to fetch JPM data: {e}")

'''    # 2. VIX 指数
    try:
        vix_df = fetch_yfinance_data(config.VIX_TICKER, config.START_DATE, config.END_DATE)
        if not vix_df.empty:
            save_raw_data(vix_df, "VIX_raw.csv")
        else:
            print("Warning: VIX data is empty.")
    except Exception as e:
        print(f"Failed to fetch VIX data: {e}")

    # 3. 国债利率（FRED）—— 保持不变
    try:
        # 注意：FRED_API_KEY 需在环境变量或 config 中设置
        rate_df = pdr.DataReader(config.FRED_SERIES_ID, 'fred', config.START_DATE, config.END_DATE, api_key=config.FRED_API_KEY)
        if not rate_df.empty:
            rate_df.rename(columns={config.FRED_SERIES_ID: 'Treasury_Rate'}, inplace=True)
            save_raw_data(rate_df, f"{config.FRED_SERIES_ID}_raw.csv")
        else:
            print("Warning: Rate data is empty.")
    except Exception as e:
        print(f"Failed to fetch rate data: {e}")'''


if __name__ == "__main__":
    main()