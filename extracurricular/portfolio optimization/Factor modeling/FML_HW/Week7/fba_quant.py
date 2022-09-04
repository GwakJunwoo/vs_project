import os
import pandas as pd
from utils import _update_stock_universe, _str_to_numeric, _update_price

class DataReader:
    
    def __init__(self, start_date, end_date, rank_criteria = "Market Cap", top_n = 30, exchange = None):
        
        self.start_date = start_date
        self.end_date = end_date
        # self.time_frame = time_frame # "Daily"
        # self.criteria = criteria # [{"Market Cap": [0, 2.120000e+12]}, {"Dividend Yield": [0, 14900]}]
        self.rank_criteria = rank_criteria # "Market Cap"
        self.top_n = top_n
        self.exchange = exchange
        self.tickers = []
    
    def get_stock_universe(self):
        print("Stock Universe Data Loading")
        if "data" not in os.listdir():
            os.mkdir("data")
        if "stock_universe.csv" not in os.listdir("data"):
            # _update_stock_universe(self.criteria) # scraping screener data is needed
            pass
        stock_universe = pd.read_csv("data/stock_universe.csv")
        
        stock_universe["Last"] = stock_universe["Last"].apply(_str_to_numeric) # This line will be removed
        stock_universe["Chg. %"] = stock_universe["Chg. %"].apply(_str_to_numeric) # This line will be removed
        stock_universe["Market Cap"] = stock_universe["Market Cap"].apply(_str_to_numeric) # This line will be removed
        stock_universe["Vol."] = stock_universe["Vol."].apply(_str_to_numeric) # This line will be removed
        
        return stock_universe 

    def get_universe_tickers(self):
        stock_universe = self.get_stock_universe().sort_values(by=[self.rank_criteria], ascending=False)
        self.tickers = stock_universe["Symbol"][:self.top_n].tolist()
        
        return self.tickers
    
    def get_price(self, field = "Close", tickers = None):
        
        if "data" not in os.listdir():
            os.mkdir("data")
        if "price" not in os.listdir("data"):
            os.mkdir("data/price")
        if not tickers:
            self.get_universe_tickers()
        
        print("Stock Price Data Loading")
        price = dict()
        for ticker in self.tickers:
            if  ticker + ".csv" not in os.listdir("data/price/"):
                _update_price([ticker], self.start_date, self.end_date)
            
            if len(pd.read_csv("data/price/" + ticker + ".csv")) <= 0:
                    self.tickers.remove(ticker)
                    print(f"{ticker} removed from the universe_tickers")
                    continue
                    
            index, data = pd.read_csv("data/price/" + ticker + ".csv")["Date"].values, \
                          pd.read_csv("data/price/" + ticker + ".csv")[field].values
            
            if self.start_date not in index or self.end_date not in index:
                _update_price([ticker], self.start_date, self.end_date)
                index, data = pd.read_csv("data/price/" + ticker + ".csv")["Date"].values, \
                              pd.read_csv("data/price/" + ticker + ".csv")[field].values
                
            price[ticker] = pd.Series(data = data, index = index)
        
        price = pd.concat([price[ticker] for ticker in self.tickers], axis = 1)
        price.columns = self.tickers

        return price
