# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
# import plotly.express as px
import pandas as pd

from src.plot_utils import plot_ohlc, plot_ts
from src.plot_utils import plot_groupby_ts
from src.utils import gen_trading_dates, get_performance_data, insert_open_prices
from src.query_utils import get_df_from_s3

external_stylesheets = [dbc.themes.BOOTSTRAP] #['https://codepen.io/chriddyp/pen/bWLwgP.css']
apt_capital_logo_filename = 'APTCapitalLogo_200x200.png' # replace with your own image

N = 10

def dash_app():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # apt_capital_logo = base64.b64encode(open(apt_capital_logo_filename, 'rb').read())

    today, signal_date, pnl_month_start, pnl_month_end = gen_trading_dates()
    trade_universe_df, open_price_df, pnl_df = get_performance_data(signal_date, pnl_month_start, pnl_month_end)
    pnl_df = insert_open_prices(pnl_df, open_price_df)

    date_range = [signal_date, min(pnl_month_end, today)]

    portfolio_plot_data = prepare_performance_df(pnl_df, trade_universe_df, N, date_range)
    performance_fig = plot_groupby_ts(portfolio_plot_data, x_col='date', y_col = 'percent_returns', g_col = 'symbol')
    
    universe_plot_data = prepare_universe_df(pnl_df, trade_universe_df, N)
    universe_top_N_fig = plot_groupby_ts(universe_plot_data, x_col='date', y_col = 'percent_return', g_col = 'symbol')

    universe_table_columns, universe_table_data = prepare_universe_table_data(pnl_df, trade_universe_df, N)

    portfolio_table_columns, portfolio_table_data = prepare_portfolio_table_data(portfolio_plot_data)

    # corr_table_columns, corr_table_data = prepare_table(signal_vol_ret_corr(all_sym_df))
    # pnl_table_columns, pnl_table_data = prepare_table(pnl_table(all_sym_df))

    app.layout = html.Div(children=[

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        html.Img(src=app.get_asset_url(apt_capital_logo_filename))
                            ), # close col 1
                    dbc.Col([
                        html.H1(children='APTCapital ASX Performance'),
                        html.Div(children = f'Performance for period {date_range[0]} to {date_range[1]}'),
                            ]), # close col 2
                        ]),
                    ]),
                    dbc.Col()
                ]), # end heading row
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        id='portfolio-graph',
                        figure=performance_fig)
                        ), # close col 1
                dbc.Col(
                    dcc.Graph(
                        id='universe-graph',
                        figure=universe_top_N_fig)
                        ) # close col 2
            ]), # close row 1
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(id='protfolio_performance',
                            columns=portfolio_table_columns,
                            data=portfolio_table_data)
                        ), # close col 1
                dbc.Col(
                    dash_table.DataTable(id='universe_performance',
                            columns=universe_table_columns,
                            data=universe_table_data)
                        ), # close col 2
            
            ]) # close row 2
    ]) # close page

    return(app)


def prepare_table(df):
    table_columns = [{"name": i, "id": i} for i in df.columns]
    table_data = df.to_dict('records')
    return(table_columns, table_data)


def prepare_performance_df(pnl_df, trade_universe_df, N, date_range):
    plot_data = pnl_df.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    plot_data['symbol'] = '#PORT_NO_STOP'
    # performance_fig_no_stop = plot_ts(plot_data, x_col='date', y_col = 'percent_returns', log=False)
    
    plot_data_stop = pnl_df.groupby(['date']).stopped_return.mean().to_frame('percent_returns').reset_index()
    plot_data_stop['symbol'] = '#PORT_TRAILING_STOP'

    topN_data = filter_to_top_N(pnl_df, trade_universe_df, N = 10)

    topN_plot_data = topN_data.groupby(['date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data['symbol'] = f'#TOP {N} PORT_NO_STOP'
    # performance_fig_no_stop = plot_ts(plot_data, x_col='date', y_col = 'percent_returns', log=False)
    
    topN_plot_data_stop = topN_data.groupby(['date']).stopped_return.mean().to_frame('percent_returns').reset_index()
    topN_plot_data_stop['symbol'] = f'#TOP {N} PORT_TRAILING_STOP'

    benchmark_data = get_benchmark_data(date_range)

    plot_data = pd.concat([plot_data, plot_data_stop, topN_plot_data, topN_plot_data_stop, benchmark_data])
    plot_data.sort_values('date', inplace = True)
    print(plot_data)

    return(plot_data)


def prepare_universe_df(pnl_df, trade_universe_df, N = 10):
    plot_data = filter_to_top_N(pnl_df, trade_universe_df, N)
    agg_data = plot_data.groupby(['date', 'open_date']).stopped_return.mean().to_frame('percent_return').reset_index()
    agg_data['symbol'] = '#PORT_TRAILING_STOP'
    plot_data['percent_return'] = plot_data.stopped_return
    return(pd.concat([plot_data, agg_data]))


def filter_to_top_N(pnl_df, trade_universe_df, N = 10):
    topN = trade_universe_df.tail(N).symbol.values
    topN_data = pnl_df.loc[pnl_df.symbol.isin(topN)].copy() #.groupby(['date', 'open_date']).percent_return.mean().to_frame('percent_returns').reset_index()
    topN_data.sort_values('date', inplace = True)
    return(topN_data)


def prepare_universe_table_data(pnl_df, trade_universe_df, N):
    df = filter_to_top_N(pnl_df, trade_universe_df, N)
    df = df.groupby('symbol')[['close', 'volume', 'high_water_mark',
                               'historical_vol', 'stop_level',
                               'stopped', 'stopped_return']].last().reset_index().round(4).sort_values('stopped')
    return(prepare_table(df))


def prepare_portfolio_table_data(df):
    df = pd.concat([
        # df.groupby('symbol').percent_returns.first().to_frame('open'),
                    df.groupby('symbol').percent_returns.last().to_frame('current'),
                    df.groupby('symbol').percent_returns.min().to_frame('min'),
                    df.groupby('symbol').percent_returns.max().to_frame('max')
                    ], axis = 1).round(4).reset_index()
    return(prepare_table(df))


def get_benchmark_data(date_range):
    index_df = get_df_from_s3('benchmark_indices')
    index_df.sort_values('timestamp', inplace = True)
    index_df = index_df.groupby(['date', 'symbol']).tail(1)
    mask = (index_df.date >= date_range[0]) & (index_df.date <= date_range[1])
    benchmark_data = index_df.loc[mask]
    benchmark_data['return'] = benchmark_data.groupby('symbol').close.pct_change()
    benchmark_data.fillna(0, inplace = True)
    benchmark_data['percent_returns'] = 1 + benchmark_data['return']
    benchmark_data['percent_returns'] = benchmark_data.groupby('symbol').percent_returns.cumprod() -1
    return(benchmark_data[['date', 'symbol', 'percent_returns']])