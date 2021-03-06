# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import json
from datetime import datetime

import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
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
stratergy = 'idio_mean_frog_all'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    html.Img(src=app.get_asset_url(apt_capital_logo_filename))
                        ), # close col 1
                dbc.Col([
                    html.H1(children='APTCapital Science'),
                    html.H1(children='Asx Performance Dashboard'),
                        ]
                    , style={'marginBottom': 2, 'marginTop': 25, 'marginLeft':5, 'marginRight':15}
                        ), # close col 2
                    ]),
                ]),
            ], style={'marginBottom': 2, 'marginTop': 5, 'marginLeft':15, 'marginRight':15}), # end heading row
    dbc.Row(
        html.H3(children='Current Month Performance')
        , style={'marginBottom': 1, 'marginTop': 1, 'marginLeft':50, 'marginRight':15} # end heading row
    ),
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
        ,
    dbc.Row(
        html.H3(children='Backtest and Historical Performance')
        , style={'marginBottom': 1, 'marginTop': 1, 'marginLeft':50, 'marginRight':15} # end heading row
    ),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='backtest-graph')
                , style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20}
            ), # close col 1
            dbc.Col(
                    [# insert backtest metrics or last few months
                    dash_table.DataTable(id='historic-performance-table')
                    , html.H6(children='Note: no leverage was used to generate the returns shown')
                    , html.H6(children='The Kelly Leverage is the maximum leverage according to the Kelly criteria')
                    ]
                    , style={'marginBottom': 5, 'marginTop': 50, 'marginLeft':20, 'marginRight':20}
            ), # close col 2
        ], style={'marginBottom': 2, 'marginTop': 5, 'marginLeft':15, 'marginRight':15}) # close row 3
    ,
    dbc.Row(
        html.H6(
            children='This website and the information contained within is for general information purposes only. It is not a source of legal, financial or investment advice. For legal, financial or investment advice consult a qualified and registered practitioner.'
            , style={'marginBottom': 1, 'marginTop': 1, 'marginLeft':50, 'marginRight':15} # end heading row

        )
    )

        , html.Div(id='hidden-data', style={'display': 'none'})
        , dcc.Dropdown(id='data-refresh', style={'display': 'none'}),
        ]) # close page


@app.callback(Output('hidden-data', 'children'), [Input('data-refresh', 'value')])
def get_data(n_clicks):
    return(get_all_dash_datasets(N))


@app.callback(Output('portfolio-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified_data):
    portfolio_df = pd.read_json(jsonified_data[0])
    performance_fig = plot_groupby_ts(portfolio_df,
                                      x_col = 'date',
                                      y_col = 'cumulative_percent_return',
                                      g_col = 'symbol',
                                      title = 'Portfolio vs benchmark (XJT) cumulative returns',
                                      yaxis_title = 'Cumulative returns on $1',
                                      log_offset=0)
    return performance_fig


@app.callback(Output('universe-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified_data):
    universe_plot_df = pd.read_json(jsonified_data[1])
    universe_plot_df = universe_plot_df.loc[universe_plot_df.stratergy==stratergy]
    universe_top_N_fig = plot_groupby_ts(universe_plot_df,
                                         x_col = 'date',
                                         y_col = 'cumulative_percent_return',
                                         g_col = 'symbol',
                                         title = 'Stratergy selected assests cumulative returns',
                                         yaxis_title = 'Cumulative returns on $1',
                                         log_offset=0)
    return universe_top_N_fig


@app.callback(Output('backtest-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified_data):
    backtest_plot_df = pd.read_json(jsonified_data[2])
    backtest_fig = plot_groupby_ts(backtest_plot_df,
                                         x_col = 'date',
                                         y_col = 'portfolio_cpnl',
                                         g_col = 'symbol',
                                         title = 'Backtest performance vs benchmark (XJT) cumulative returns',
                                         yaxis_title = 'Cumulative returns on $1',
                                         log_offset=0,
                                         log=True)
    historic_df = pd.read_json(jsonified_data[3])
    oos_date = historic_df.loc[historic_df.data_split == 'validation', 'start_year'].values[0]
    oos_date = datetime.strptime(f'01-02-{oos_date}', '%d-%m-%Y')
    backtest_fig.add_trace(go.Scatter(x=[oos_date,oos_date],
                                      y=[backtest_plot_df.portfolio_cpnl.min(), backtest_plot_df.portfolio_cpnl.max()],
                                      mode='lines',
                                      name='Data split',
                                      line=dict(
                                          dash="dot")
                                      ))

    return(backtest_fig)


@app.callback([Output('historic-performance-table', 'columns'), Output('historic-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    cols = ['stratergy', 'stratergy_period', 'N', 'start_year', 'end_year', 'final_wealth_multiple', 'sharpe_ratio', 'kelly_leverage', 'data_split']
    historic_df = prepare_table(pd.read_json(jsonified_data[3])[cols])
    return historic_df


@app.callback([Output('universe-performance-table', 'columns'), Output('universe-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    universe_df = prepare_universe_table_data(pd.read_json(jsonified_data[1]), stratergy)
    return universe_df


@app.callback([Output('protfolio-performance-table', 'columns'), Output('protfolio-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    portfolio_df = prepare_portfolio_table_data(pd.read_json(jsonified_data[0]))
    return portfolio_df