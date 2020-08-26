import json

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from tabs import individual_components_page, portfolio_page
from tabs import next_period

from src.dash_utils import *

from src.utils import gen_trading_dates, get_performance_data
from src.utils import insert_open_prices, get_current_date_tz
from src.query_utils import get_df_from_s3


external_stylesheets = [dbc.themes.BOOTSTRAP] #['https://codepen.io/chriddyp/pen/bWLwgP.css']
apt_capital_logo_filename = 'APTCapitalLogo_200x200.png' # replace with your own image

N=10


app.layout = html.Div(children=[
    dbc.Row([
        dbc.Col([
            html.Img(src=app.get_asset_url(apt_capital_logo_filename)),
        ]), # end col 2
        dbc.Col([
            html.H1(children='APTCapital Asx Performance Dashboard'),
            html.H3(id='data-refresh'),
            html.Button('Refresh Data', id='button'),
            html.H3(id='button-clicks'),
        ])

    ], style={'marginBottom': 50, 'marginTop': 25, 'marginLeft':15, 'marginRight':15}), # end row 1
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="tabs",
                    value='tab-1',
                    children=[
                        dcc.Tab(label='Portfolio', value='Portfolio'),
                        dcc.Tab(label='Individual components', value='Individual'),
                        dcc.Tab(label='Next Period', value='Next_period')
                        ]
                    ),
                    html.Div(id='tabs-content'),
        ])
    ], style={'marginBottom': 50, 'marginTop': 25, 'marginLeft':15, 'marginRight':15}), # end row 2 (main page )
    html.Div(id='hidden-data', style={'display': 'none'})
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def display_page(tab):
    if tab == 'Portfolio':
        return portfolio_page.layout
    elif tab == 'Individual':
        return individual_components_page.layout
    elif tab == 'Next_period':
        return next_period.layout
    else:
        return 'Please select a tab above'


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

    # next trading period data
    next_signal_df, next_trade_universe_df = next_period_signal(pnl_month_end)

    out = [portfolio_plot_data.to_json(),
            universe_plot_data.to_json(),
            pnl_df.to_json(),
            trade_universe_df.to_json(),
            json.dumps(str(last_update)[:16]),
            json.dumps(N),
            next_signal_df.to_json(),
            next_trade_universe_df.to_json()]

    print(f'len out json {len(out)}')

    return(out)