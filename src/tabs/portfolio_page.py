import json

import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

from src.app import app
from src.plot_utils import plot_groupby_ts
from src.dash_utils import *


layout = html.Div([
    html.H3('Portfolio Performance'),
    dcc.Graph(id='portfolio-graph'),
    dash_table.DataTable(id='protfolio-performance-table')
])

@app.callback(Output('portfolio-graph', 'figure'), [Input('hidden-data', 'children'),
                                                    Input('button', 'n_clicks')])
def render_graph(jsonified__data, refresh):
    portfolio_df = pd.read_json(jsonified__data[0])
    performance_fig = plot_groupby_ts(portfolio_df, x_col='date', y_col = 'percent_returns', g_col = 'symbol', title = f'graph update {json.loads(jsonified__data[4])}')
    return performance_fig


@app.callback([Output('protfolio-performance-table', 'columns'), Output('protfolio-performance-table', 'data')], [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    # universe_df = prepare_universe_table_data(pd.read_json(jsonified__data[2]), pd.read_json(jsonified__data[3]), N)
    portfolio_df = prepare_portfolio_table_data(pd.read_json(jsonified_data[0]))
    return portfolio_df