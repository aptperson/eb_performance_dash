# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import json

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.express as px
import pandas as pd

from src.dash_utils import *

from src.plot_utils import plot_ohlc, plot_ts
from src.plot_utils import plot_groupby_ts
from src.utils import gen_trading_dates, get_performance_data
from src.utils import insert_open_prices, get_current_date_tz
from src.query_utils import get_df_from_s3

external_stylesheets = [dbc.themes.BOOTSTRAP]
apt_capital_logo_filename = 'APTCapitalLogo_200x200.png'

N = 10
stratergy = 'frog_agg'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    html.Img(src=app.get_asset_url(apt_capital_logo_filename))
                        ), # close col 1
                dbc.Col([
                    html.H1(children='APTCapital Asx Performance Dashboard'),
                        ]
                    , style={'marginBottom': 2, 'marginTop': 25, 'marginLeft':5, 'marginRight':15}
                        ), # close col 2
                dbc.Col([
                    html.H6(id='data-refresh'),
                    html.Button('Refresh Data', id='button'),
                    # html.Div(children = f'Performance for period {date_range[0]} to {date_range[1]}'),
                        ]
                    , style={'marginBottom': 2, 'marginTop': 50, 'marginLeft':50, 'marginRight':15}
                        ), # close col 3
                    ]),
                ]),
                # dbc.Col()
            ], style={'marginBottom': 2, 'marginTop': 5, 'marginLeft':15, 'marginRight':15}), # end heading row
    dbc.Row([
            dbc.Col(
                dcc.Graph(id='portfolio-graph')
                , style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20}
                ), # close col 1
            dbc.Col(
                dcc.Graph(id='universe-graph')
                , style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20}
                    ) # close col 2
        ])
        , # close row 1
        dbc.Row([
            dbc.Col(
                dash_table.DataTable(id='protfolio-performance-table')
                , style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20}
            ), # close col 1
            dbc.Col(
                dash_table.DataTable(id='universe-performance-table')
                , style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20}
            ), # close col 2
        ], style={'marginBottom': 2, 'marginTop': 5, 'marginLeft':15, 'marginRight':15}) # close row 2
        , html.Div(id='hidden-data', style={'display': 'none'})
        ]) # close page


@app.callback(Output('hidden-data', 'children'), [Input('button', 'n_clicks')])
def get_data(n_clicks):
    print(f'N data updates: {n_clicks}')
    last_update = get_current_date_tz(out_format=None)
    today, signal_date, pnl_month_start, pnl_month_end = gen_trading_dates()
    trade_universe_df, open_price_df, pnl_df = get_performance_data(signal_date, pnl_month_start, pnl_month_end)
    pnl_df = insert_open_prices(pnl_df, open_price_df)
    date_range = [signal_date, min(pnl_month_end, today)]
    portfolio_plot_data = prepare_performance_df(pnl_df, trade_universe_df, N, date_range)
    portfolio_plot_data.reset_index(drop = True, inplace = True)

    universe_plot_data = prepare_universe_df(pnl_df, trade_universe_df, N)
    universe_plot_data.reset_index(drop = True, inplace = True)

    return [portfolio_plot_data.to_json(),
            universe_plot_data.to_json(),
            pnl_df.to_json(),
            trade_universe_df.to_json(),
            json.dumps(str(last_update)[:16])]


@app.callback(Output('data-refresh', 'children'), [Input('hidden-data', 'children')])
def timestamp_text(timestamp):
    return f'The last data refresh was at AEST: {json.loads(timestamp[4])}'


@app.callback(Output('portfolio-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified_data):
    portfolio_df = pd.read_json(jsonified_data[0])
    performance_fig = plot_groupby_ts(portfolio_df,
                                      x_col = 'date',
                                      y_col = 'percent_returns',
                                      g_col = 'symbol',
                                      title = 'Portfolio vs benchmark (XJT) returns',
                                      yaxis_title = 'Percent Return (%)')
    return performance_fig


@app.callback(Output('universe-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified_data):
    universe_plot_df = pd.read_json(jsonified_data[1])
    universe_plot_df = universe_plot_df.loc[universe_plot_df.stratergy==stratergy]
    universe_top_N_fig = plot_groupby_ts(universe_plot_df,
                                         x_col = 'date',
                                         y_col = 'percent_return',
                                         g_col = 'symbol',
                                         title = 'Momentum with frog filter portfolio component returns',
                                         yaxis_title = 'Percent Return (%)')
    return universe_top_N_fig


@app.callback([Output('universe-performance-table', 'columns'), Output('universe-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    universe_df = prepare_universe_table_data(pd.read_json(jsonified_data[1]), stratergy)
    return universe_df


@app.callback([Output('protfolio-performance-table', 'columns'), Output('protfolio-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    # universe_df = prepare_universe_table_data(pd.read_json(jsonified_data[2]), pd.read_json(jsonified_data[3]), N)
    portfolio_df = prepare_portfolio_table_data(pd.read_json(jsonified_data[0]))
    return portfolio_df