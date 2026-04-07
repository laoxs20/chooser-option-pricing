import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 1. 定义标的资产
ticker = 'JPM'

# 2. 获取所有可用的期权到期日
stock = yf.Ticker(ticker)
expirations = stock.options
print(f"Available expiration dates: {expirations}")

# 3. 计算并找到最接近1年（365天）的到期日
target_date = datetime.now() + timedelta(days=365)
closest_expiry = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
print(f"Closest expiry to 1 year: {closest_expiry}")

# 4. 获取该到期日的期权链
opt_chain = stock.option_chain(date=closest_expiry)

# 5. 分别查看看涨和看跌期权的数据
calls = opt_chain.calls
puts = opt_chain.puts
print(calls.head())
print(puts.head())

# 筛选出 strike = 150 的看涨期权数据
target_strike = 150
specific_call = calls[calls['strike'] == target_strike]
specific_put = puts[puts['strike'] == target_strike]

print(f"Call option data for strike {target_strike}:")
print(specific_call)
print(f"Put option data for strike {target_strike}:")
print(specific_put)