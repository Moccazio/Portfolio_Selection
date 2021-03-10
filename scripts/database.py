import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, ticker, start=None, end=None):

        self.ticker = ticker

        try:
            self._ticker = yf.Ticker(self.ticker)

            if not (start or end):
                self.df = self.df_ = self._ticker.history(period='max', auto_adjust=True)
                
            else:
                self.df = self.df_ = self._ticker.history(start=start, end=end, auto_adjust=True)

        except Exception as err:
            print(err)