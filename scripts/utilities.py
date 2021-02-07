from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps

# ========================================
# General Utilities
# ========================================

def to_return(prices,ret='simple'):
    if ret == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret


def cprint(df, nrows=None):
    if not isinstance(df, (pd.DataFrame,)):
        try:
            df = df.to_frame()
        except:
            raise ValueError("object cannot be coerced to df")

    if not nrows:
        nrows = 5
    print("-" * 79)
    print("dataframe information")
    print("-" * 79)
    print(df.tail(nrows))
    print("-" * 50)
    print(df.info())
    print("-" * 79)
    print()

def fill_na(df):
    df = df.transform(lambda x: x.fillna(x.mean()))
    return df

def get_column_range(series):
    """
    get min and max values
    :param series: array-like
    :return: float, float
    """
    return min(series), max(series)


def make_ic_series(list_of_tups, name=None):
    """fn: convert list of tuples for
            information criterion (aic, bic) into series
    # args
        list_of_tups : list() of tuples()
            tuple[0] is n_component, tuple[1] is IC
        name : str(), name of IC
    # returns
        s : pd.Series()
            index is n_components, values are IC's
    """
    s = (
        pd.DataFrame(list_of_tups)
        .rename(columns={0: "n_components", 1: name})
        .set_index("n_components")
        .squeeze()
    )
    return s

import re
import decorator
import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle


def adjust(date, close, adj_close, in_col, rounding=4):
    try:
        factor = adj_close / close
        return round(in_col * factor, rounding)
    except ZeroDivisionError:
        print('WARNING: DIRTY DATA >> {} Close: {} | Adj Close {} | in_col: {}'.format(date, close, adj_close, in_col))
        return 0   
    
def clean_ticker(ticker):
    """
    Cleans a ticker for easier use throughout MoneyTree
    Splits by space and only keeps first bit. Also removes
    any characters that are not letters. Returns as lowercase.
    >>> clean_ticker('^VIX')
    'vix'
    >>> clean_ticker('SPX Index')
    'spx'
    """
    pattern = re.compile('[\W_]+')
    res = pattern.sub('', ticker.split(' ')[0])
    return res.lower()


def clean_tickers(tickers):
    """
    Maps clean_ticker over tickers.
    """
    return [clean_ticker(x) for x in tickers]


def fmtp(number):
    """
    Formatting helper - percent
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2%')


def fmtpn(number):
    """
    Formatting helper - percent no % sign
    """
    if np.isnan(number):
        return '-'
    return format(number * 100, '.2f')


def fmtn(number):
    """
    Formatting helper - float
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2f')


def get_freq_name(period):
    period = period.upper()
    periods = {
        'B': 'business day',
        'C': 'custom business day',
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'BM': 'business month end',
        'CBM': 'custom business month end',
        'MS': 'month start',
        'BMS': 'business month start',
        'CBMS': 'custom business month start',
        'Q': 'quarterly',
        'BQ': 'business quarter end',
        'QS': 'quarter start',
        'BQS': 'business quarter start',
        'Y': 'yearly',
        'A': 'yearly',
        'BA': 'business year end',
        'AS': 'year start',
        'BAS': 'business year start',
        'H': 'hourly',
        'T': 'minutely',
        'S': 'secondly',
        'L': 'milliseonds',
        'U': 'microseconds'}

    if period in periods:
        return periods[period]
    else:
        return None


def scale(val, src, dst):
    """
    Scale value from src range to dst range.
    If value outside bounds, it is clipped and set to
    the low or high bound of dst.
    Ex:
        scale(0, (0.0, 99.0), (-1.0, 1.0)) == -1.0
        scale(-5, (0.0, 99.0), (-1.0, 1.0)) == -1.0
    """
    if val < src[0]:
        return dst[0]
    if val > src[1]:
        return dst[1]

    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


def as_percent(self, digits=2):
    return as_format(self, '.%s%%' % digits)


def as_format(item, format_str='.2f'):
    """
    Map a format string over a pandas object.
    """
    if isinstance(item, pd.Series):
        return item.map(lambda x: format(x, format_str))
    elif isinstance(item, pd.DataFrame):
        return item.applymap(lambda x: format(x, format_str))
    
# ================================================
# Print it Custom 
# ================================================
def custom_describe(df, nidx=3, nfeats=20):
    ''' Concat transposed topN rows, numerical desc & dtypes '''

    print(df.shape)
    nrows = df.shape[0]
    
    rndidx = np.random.randint(0,len(df),nidx)
    dfdesc = df.describe().T

    for col in ['mean','std']:
        dfdesc[col] = dfdesc[col].apply(lambda x: np.round(x,2))
 
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), axis=1, join='outer')
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)
    
    # add count nonNAN, min, max for string cols
    nan_sum = df.isnull().sum()
    dfout['count'] = nrows - nan_sum
    dfout['min'] = df.min().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['nunique'] = df.apply(pd.Series.nunique)
    dfout['nan_count'] = nan_sum
    dfout['pct_nan'] = nan_sum / nrows
    
    return dfout.iloc[:nfeats, :]

# ================================================
