import yfinance as yf
import pandas as pd

# 定义需要获取的外汇数据，使用HKD=X（香港元对美元的汇率）
ticker = "HKD=X"

# 下载过去60天的数据
data = yf.download(ticker, period="2y", interval="1d")

# 检查数据结构
print("Raw Data Head:\n", data.head())

# 将数据列名重命名，以符合预期格式
data.reset_index(inplace=True)  # 把日期从索引变为普通列
data.rename(columns={"Date": "Date", "Open": "Price", "Adj Close": "ExchangeRate", "Close": "Close",
                     "High": "High", "Low": "Low", "Volume": "Volume"}, inplace=True)

# 保存数据到CSV文件
data.to_csv("hkd_exchange_rate.csv", index=False)

# 输出已保存的CSV文件路径
print("CSV文件已保存：hkd_exchange_rate.csv")