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
    html.H3('Next Period Components'),
    dcc.Graph(id='symbol-graph'),
    dash_table.DataTable(id='symbol-signal-table',
                         row_selectable="single")
])

# sym = 'BRN'
# sym_df = signal_df.loc[signal_df.symbol == sym]
# print(sym_df.head())
# print(sym_df.close.values[-1] / sym_df.close.values[0] - 1)

# plot_data = pd.melt(sym_df, value_vars = ['raw_close', 'close'], id_vars = ['date', 'symbol'])
# plot_groupby_ts(plot_data, 'date', 'value', 'variable')


@app.callback([Output('symbol-signal-table', 'columns'),
               Output('symbol-signal-table', 'data')],
               [Input('hidden-data', 'children')])
def render_table(jsonified_data):
    next_trade_universe_df = pd.read_json(jsonified_data[7])
    N = int(jsonified_data[5])
    table_data = prepare_next_period_universe_table_data(next_trade_universe_df, N)
    return(table_data)


@app.callback(Output('symbol-graph', 'figure'),
              [Input('hidden-data', 'children'),
              Input('symbol-signal-table', 'selected_rows'),
              Input('symbol-signal-table', 'data')])
def render_graph(jsonified_data, selected_rows, table_data):
    next_signal_df = pd.read_json(jsonified_data[6])
    if selected_rows is None:
        sym = next_signal_df.symbol.values[-1]
    else:
        sym = table_data[selected_rows[0]]['symbol']
    print(f'selected symbol {sym}')
    sym_df = next_signal_df.loc[next_signal_df.symbol == sym]
    plot_data = pd.melt(sym_df, value_vars = ['raw_close', 'close'], id_vars = ['date', 'symbol'])
    symbol_raw_close_fig = plot_groupby_ts(plot_data, x_col='date', y_col='value', g_col='variable')
    return symbol_raw_close_fig