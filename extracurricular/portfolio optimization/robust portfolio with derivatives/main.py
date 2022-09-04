import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import cvxpy as cvx
from abc import abstractmethod
from abc import ABC
from typing import Dict, Optional, Tuple, List, TypeVar, Union
from pykrx import stock

"""파생상품을 포함한 견고한 포트폴리오 최적화 알고리즘"""

T = TypeVar('T')

"""
class Base(ABC):
    Base class for all financial instruments

    @abstractmethod
    def returns(self, df: pd.DataFrame):
        Shares, ETF: daily stock return
        Options: the tuple(a, B), option returns = f(base_stock_return: x) = max(0, a + Bx)
        ELW: it will return the tuple set such as Options.
"""


class Portfolio:
    def __init__(self, risk_aversion):
        """
        n: The number of stock
        m: The number of derivatives
        L: length of time-series
        buffer: a dictionary that stores all the data needed for the operation
        """

        self.buffer = dict()
        self.reference_price = np.array([])  # stock daily price dataframe (n x 1)
        self.stock_return = np.array([])  # stock daily return matrix (n x L-1)
        self.sigma = np.array([])  # stock daily return covariance matrix (n x n)
        self.mu = np.array([])  # stock return mean matrix (n x 1)
        self.weight_stock = np.array([])  # stock optimal weights matrix (n x 1)
        self.weight_derivative = np.array([])  # derivative optimal weights matrix (m x 1)
        self.B = np.array([])  # derivative <-> base stock mapping & parameter matrix
        self.a = np.array([])  # derivative return function parameter matrix
        self.y = np.array([])  # variable matrix for optimization (m x 1)
        self.s = np.array([])  # variable matrix for optimization (n x 1)
        self.p = risk_aversion  # risk-aversion 0<=p<=1
        self.xi = np.sqrt(self.p / (1 - self.p))  # adjusted risk aversion
        self.ticker_list = list()
        self.derivative_list = list()
        self.n = 0
        self.m = 0

    def add(self, security):

        if security.buffer['type'] == 'stock':
            if self.reference_price.size != 0 and self.stock_return.size != 0:
                self.reference_price = np.vstack((self.reference_price, security.buffer['reference_price']))
                self.stock_return = np.vstack((self.stock_return, security.buffer['daily_return']))
            else:
                self.reference_price = np.append(self.reference_price, security.buffer['reference_price'])
                self.stock_return = np.append(self.stock_return, security.buffer['daily_return'])
            self.ticker_list.append(security.buffer['ticker'])

            try:
                if self.B.size != 0:
                    self.B = np.hstack((self.B, np.zeros((len(self.derivative_list), 1))))
                else:
                    self.B = np.zeros((self.m, self.n))
            except Exception as e:
                print(e)

            self.n += 1
            print("Stock {} is successfully inserted".format(security.buffer['ticker']))

        # TODO
        elif security.buffer['type'] == 'derivative':
            """
            self.B =
            self.a =
            """
            self.m += 1
            self.derivative_list.append(security.buffer['name'])
            if self.a.size != 0 and self.B.size != 0:
                self.a = np.vstack((self.a, security.buffer['a']))
                self.B = np.vstack((self.B, np.zeros((1, len(self.ticker_list)))))
            else:
                self.a = np.append(self.a, security.buffer['a'])
                self.B = np.zeros((self.m, self.n))

            tmp = self.ticker_list.index(security.buffer['base_ticker'])
            row = self.B.shape[0] - 1
            self.B[row, tmp] = security.buffer['B']

            print("Derivatives {} is successfully inserted".format(security.buffer['name']))

    def fit(self, detail=False):
        self.buffer['stock_price'] = self.reference_price
        self.buffer['stock_return'] = self.stock_return
        self.mu = np.mean(self.buffer['stock_return'], axis=1).T
        self.sigma = np.cov(self.buffer['stock_return'])
        self.buffer['mu'] = self.mu
        self.buffer['sigma'] = self.sigma
        self.buffer['B'] = self.B
        self.buffer['a'] = self.a
        self.buffer['xi'] = self.xi

        n = self.buffer['mu'].shape[0]
        m = self.buffer['B'].shape[0]

        w = cvx.Variable((n, 1))
        wd = cvx.Variable((m, 1))
        y = cvx.Variable((m, 1))
        s = cvx.Variable((n, 1))

        w_one = np.ones((n, 1))
        wd_one = np.ones((m, 1))

        st1 = self.buffer['mu'].T @ (w + self.buffer['B'].T @ y - s)
        st2 = self.buffer['xi'] * cvx.norm2(np.sqrt(self.buffer['sigma']) @ (w + self.buffer['B'].T @ y - s))
        st3 = self.buffer['a'].T @ y

        try:
            obj = cvx.Maximize(st1 - st2 + st3)
            constraints = [y <= wd, -y <= 0, -s <= 0, w_one.T @ w + wd_one.T @ wd == 1, -w <= 0]
            prob = cvx.Problem(obj, constraints)
            result = prob.solve(solver='SCS')

        except Exception as e:
            print(e)
            result = None

        self.weight_stock = w.value
        self.weight_derivative = wd.value
        self.y = y.value
        self.s = s.value

        if detail:
            print("stock allocation: \n{} \nderivatives allocation:\n{}".format(self.weight_stock, self.weight_derivative))

        return result

    def set_risk_aversion(self, risk_aversion):
        self.p = risk_aversion
        self.xi = np.sqrt(self.p / (1 - self.p))
        self.buffer['xi'] = self.xi

    def view_parameters(self):
        for key, value in self.buffer.items():
            try:
                print('{}: {}'.format(key, value.shape))

            except Exception as e:
                print('{}: {}'.format(key, value))


class Stock:
    def __init__(self, tickers, start_date='2020-01-04', reference_date='2022-01-04'):
        self.buffer = dict()
        self.register_buffer('type', 'stock')
        daily_price = fdr.DataReader(tickers, start=start_date, end=reference_date)

        daily_return = self.returns(df=daily_price)
        reference_date = reference_date
        reference_price = daily_price.iloc[-1]['Close']

        self.register_buffer('ticker', tickers)
        self.register_buffer('daily_return', daily_return)
        self.register_buffer('reference_date', reference_date)
        self.register_buffer('reference_price', reference_price)

    def register_buffer(self, name: str, buffer):
        self.buffer[name] = buffer

    @staticmethod
    def returns(df: pd.DataFrame):
        return np.array(df['Close'].pct_change()[1:])

    def view_parameters(self):
        for key, value in self.buffer.items():
            print('{}: {}'.format(key, value))


# TODO
class Derivative:
    def __init__(self, base_stock, category, name, strike_price, premium):
        self.buffer = dict()
        self.register_buffer('type', 'derivative')
        self.register_buffer('base_ticker', base_stock.buffer['ticker'])
        self.register_buffer('base_stock', stock.get_market_ticker_name(base_stock.buffer['ticker']))
        self.register_buffer('category', category)
        self.register_buffer('name', name)
        self.register_buffer('S0', base_stock.buffer['reference_price'])
        self.register_buffer('K', strike_price)
        self.register_buffer('cost', premium)

        self.returns()

    def returns(self):
        if self.buffer['category'] == '콜' or 'call':
            self.buffer['a'] = -(self.buffer['K'] / self.buffer['cost'])
            self.buffer['B'] = (self.buffer['S0'] / self.buffer['cost'])
        elif self.buffer['category'] == '풋' or 'put':
            self.buffer['a'] = (self.buffer['K'] / self.buffer['cost'])
            self.buffer['B'] = -(self.buffer['S0'] / self.buffer['cost'])
        else:
            raise Exception('Type error!')

    def register_buffer(self, name: str, buffer):
        self.buffer[name] = buffer

    def view_parameters(self):
        for key, value in self.buffer.items():
            print('{}: {}'.format(key, value))
