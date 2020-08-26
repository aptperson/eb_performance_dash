import dash

import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP] #['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server