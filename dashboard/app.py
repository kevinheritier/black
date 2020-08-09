from __future__ import absolute_import, division, print_function

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import uuid


external_stylesheets = [dbc.themes.COSMO]

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
dash_app.config['suppress_callback_exceptions'] = True
dash_app.title = "Black-Litterman Dashboard"
# dash_app.server.secret_key = str(uuid.uuid4())  # for session data storage


dash_app.layout = html.Div(children=[
    html.Header([
        html.H2(dash_app.title, style={'color': 'black',
                                       "padding-left": "15px"}),
    ], className='row', style={
        'height': '70px',
        'margin-bottom': '50px'
    })


], style={"padding-left": "15px", "padding-right": "15px"})

if __name__ == "__main__":
    dash_app.run_server(debug=True)
