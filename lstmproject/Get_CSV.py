import yfinance as yf
import pandas as pd

ticker = "HKD=X"

data = yf.download(ticker, period="2y", interval="1d")

print("Raw Data Head:\n", data.head())

data.reset_index(inplace=True) 
data.rename(columns={"Date": "Date", "Open": "Price", "Adj Close": "ExchangeRate", "Close": "Close",
                     "High": "High", "Low": "Low", "Volume": "Volume"}, inplace=True)


data.to_csv("hkd_exchange_rate.csv", index=False)

print("CSV文件已保存：hkd_exchange_rate.csv")
