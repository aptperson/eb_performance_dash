import json

import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

from app import app
from src.plot_utils import plot_groupby_ts
from src.dash_utils import *

layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='universe-raw-graph'),
            ]),
        dbc.Col([
            dcc.Graph(id='universe-stopped-graph'),
            ]),
    ]),
    dash_table.DataTable(id='universe-performance-table')
], style={'marginBottom': 50, 'marginTop': 5, 'marginLeft':20, 'marginRight':20})

@app.callback(Output('universe-raw-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified__data):
    universe_plot_df = pd.read_json(jsonified__data[1])
    universe_top_N_fig = plot_groupby_ts(universe_plot_df, x_col='date', y_col = 'percent_return', g_col = 'symbol', title = f'No Stop Returns for Portfolio Components')
    return universe_top_N_fig


@app.callback(Output('universe-stopped-graph', 'figure'), [Input('hidden-data', 'children')])
def render_graph(jsonified__data):
    universe_plot_df = pd.read_json(jsonified__data[1])
    universe_top_N_fig = plot_groupby_ts(universe_plot_df, x_col='date', y_col = 'stopped_return', g_col = 'symbol', title = f'Returns with trailing stop for Portfolio Components')
    return universe_top_N_fig


@app.callback([Output('universe-performance-table', 'columns'),
               Output('universe-performance-table', 'data')],
               [Input('hidden-data', 'children')])
def render_table(jsonified__data):
    N = int(jsonified__data[5])
    universe_df = prepare_universe_table_data(pd.read_json(jsonified__data[2]), pd.read_json(jsonified__data[3]), N)
    return universe_df