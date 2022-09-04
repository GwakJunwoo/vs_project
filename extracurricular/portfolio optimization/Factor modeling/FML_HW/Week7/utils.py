import os
import datetime
import FinanceDataReader as fdr


def _update_stock_universe():
    pass
    
def _str_to_numeric(data):
    if data[-1] == 'T':
        return float(data[:-1]) * 10**(12)
    elif data[-1] == 'B':
        return float(data[:-1]) * 10**(9)
    elif data[-1] == 'M':
        return float(data[:-1]) * 10**(6)
    elif data[-1] == 'K':
        return float(data[:-1]) * 10**(3)
    elif data[-1] == '%':
        return float(data[:-1]) * 10**(-2)
    else:
        return float(data)

def _update_price(tickers, start_date, end_date):
        
        for ticker in tickers:
            fdr.DataReader(ticker, start_date, end_date).to_csv("data/price/" + ticker + ".csv")
