from datetime import datetime

import pandas as pd
import plotly
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

# import matplotlib
# import matplotlib.pyplot as plt
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

def plot_ret_series(ret_series, plot_vix = False, log = True, col = 'ret'):

    if isinstance(ret_series, pd.DataFrame):
        tmp = ret_series.reset_index().groupby(['year', 'month']).tail(1)
        if plot_vix:
            vix = ret_series.reset_index().groupby(['year', 'month']).max()
    else:
        tmp = ret_series.to_frame(col).reset_index().groupby(['year', 'month']).tail(1)
        if plot_vix:
            print('vix has not been passed')
            plot_vix = False
            
    x = tmp.date
    if log:
        y = np.log10(tmp[col])
    else:
        y = tmp[col]
        
    fig, ax1 = plt.subplots()
    ax1.plot(x, y)

#     plt.plot(x,y)
    
    if plot_vix:
#     if log:
#         y = np.log10(tmp.vix)
#     else:
        y2 = vix.vix
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.plot(x, y2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    
    plt.show()


def plot_ts(plot_data, y_col, x_col = 'date', date_format = '%Y-%m-%d', log = True):
    
    x = x_date_conversion(plot_data[x_col], date_format)

    y = plot_data[y_col]
    if log:
        y = np.log10(y)

    data = [dict(
      type = 'scatter',
      x = x,
      y = y,
      mode = 'lines'
    )]

    fig = go.Figure(data = data)
    return(fig)


def x_date_conversion(x, date_format):
    if isinstance(x.iloc[0], str):
        x = x.apply(datetime.strptime, args = [date_format])
    return(x)


def plot_ohlc(plot_data, x_col = 'date', date_format = '%Y-%m-%d'):
    
    x = x_date_conversion(plot_data[x_col], date_format)
    
    O = plot_data.open
    H = plot_data.high
    L = plot_data.low
    C = plot_data.close

    data = [dict(
      type = 'ohlc',
      x = x,
      open = O,
      high = H,
      low = L,
      close = C
      )]

    fig = go.Figure(data = data)
    return(fig)
    

def plot_groupby_ts(plot_data,
                    x_col,
                    y_col,
                    g_col,
                    date_format = '%Y-%m-%d',
                    title = None,
                    yaxis_title = None):
    x = x_date_conversion(plot_data[x_col], date_format)
    fig = go.Figure()
    for group, data, in plot_data.groupby(g_col):
        fig.add_scatter(x = data[x_col], y = data[y_col], name = group, mode = 'lines')
    fig.update_layout(title={'text': title},
                      xaxis={'tickformat': '%Y-%m-%d'},
                      yaxis_title={'text': yaxis_title})

    return(fig)

    
def zero_returns_by_open_date(df):
    zero_ret_dates = df.open_date.unique()
    zero_ret = np.zeros(len(zero_ret_dates))
    return(pd.DataFrame({'date':zero_ret_dates,
                         'open_date': zero_ret_dates,
                         'returns': zero_ret}))


def zero_returns_by_first_open_date(df):
    zero_ret_dates = df.groupby('symbol').open_date.first()
    zero_ret = np.zeros(len(zero_ret_dates))
    return(pd.DataFrame({'date':zero_ret_dates,
                         'open_date': zero_ret_dates,
                         'returns': zero_ret}))
    
    
def returns_by_open_date(trade_df):
    plot_data = trade_df.groupby(['date', 'open_date']).percent_ret.mean()
    plot_data = plot_data.to_frame('returns').reset_index()
    
    zero_ret_dates = plot_data.open_date.unique()
    zero_ret = np.zeros(len(zero_ret_dates))
    plot_data = pd.concat([plot_data, 
               pd.DataFrame({'date':zero_ret_dates,
                             'open_date': zero_ret_dates,
                             'returns': zero_ret})])
    plot_data.sort_values('date', inplace = True)
    plot_data['date'] = plot_data.date.apply(datetime.strptime, args = ['%Y%m%d'])
    return(plot_data)


def returns_by_symbol_open_date(trade_df):
    plot_data = trade_df.groupby(['symbol', 'date', 'open_date']).percent_ret.mean()
    plot_data = plot_data.to_frame('returns').reset_index()
    
    for sym in trade_df.symbol.unique():
        zero_ret_df = zero_returns_by_first_open_date(
            plot_data.loc[plot_data.symbol == sym])
        zero_ret_df['symbol'] = sym
        plot_data = pd.concat([plot_data, zero_ret_df])
    plot_data.sort_values('date', inplace = True)
    plot_data['date'] = plot_data.date.apply(datetime.strptime, args = ['%Y%m%d'])
    return(plot_data)