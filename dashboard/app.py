from __future__ import absolute_import, division, print_function
import logging
import io
import dash
import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import dash_table
import uuid
# from analytics import portfolio_views


external_stylesheets = [dbc.themes.COSMO]

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
dash_app.config['suppress_callback_exceptions'] = True
dash_app.title = "Black-Litterman Dashboard"
# dash_app.server.secret_key = str(uuid.uuid4())  # for session data storage


dash_app.layout = html.Div(children=[
    # Header
    html.Header([
        html.H1(dash_app.title, style={'color': 'black',
                                       "padding-left": "50px",
                                       "margin-top": "20px"}),
    ], className='row', style={
        'height': '70px',
        'margin-bottom': '20px'
    }),

    # Upload Portfolio and Views
    html.Div(children=[
        html.Div(children="Upload portfolio, covariance and views", style={'margin-left': '40px',
                                                                           'margin-top': '0px'}),
        dcc.Upload(
            id='upload-excel',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File', style={
                    "color": "blue", "text-decoration": "underline"})
            ]),
            style={"margin-left": "50px",
                   "borderStyle": "dashed",
                   "borderRadius": "5px",
                   "margin": "10px",
                   "lineHeight": "60px",
                   "height": "60px",
                   "textAlign": "center",
                   "borderWidth": "1px",
                   "width": "20%"
                   },
            multiple=False),

        # spaw children when ptf uploaded

        html.H2("Portfolio", style={'color': 'black',
                                    "padding-left": "50px",
                                    "margin-top": "20px"}),
        html.Div(id="portfolio-sheet", children=[]),
        html.Div(id="portfolio-rows", children=[]),
        html.H2("Covariance", style={'color': 'black',
                                     "padding-left": "50px",
                                     "margin-top": "20px"}),
        html.Div(id="covariance-sheet", children=[]),
        html.H2("Views", style={'color': 'black',
                                "padding-left": "50px",
                                "margin-top": "20px"}),
        html.Div(id="views-sheet", children=[]),
        html.Div(id="views-rows", children=[]),
    ]),

    # Variable caching
    html.Div([
        html.Div(
            dcc.Store(id="raw_input_dict", data=None),
            id="raw_input_dict_parent"
        )
    ])

], style={"padding-left": "15px", "padding-right": "15px"})


@dash_app.callback(
    [Output(component_id="portfolio-sheet", component_property="children"),
     Output(component_id="covariance-sheet", component_property="children"),
     Output(component_id="views-sheet", component_property="children"),
     Output(component_id="raw_input_dict", component_property="data")],
    [Input(component_id="upload-excel", component_property="contents")],
    [State(component_id="upload-excel", component_property="filename")]
)
def load_excel(data, filename):

    # if prevent_updates == "True":
    #     raise PreventUpdate

    if data and filename:
        logging.info("Loading custom portfolio...")
        content_type, content_string = data.split(',')
        decoded = base64.b64decode(content_string)
        extension = filename.split('.')[-1]
        if extension in ["xlsx", "xls"]:
            sheet_to_df_map = pd.read_excel(
                io.BytesIO(decoded), sheet_name=None, header=1, index_col=0)
        else:
            raise Exception("Unsupported file type"
                            "Please use .xls(x) file."
                            "Found: {}".format(filename))
        logging.info("Custom excel file loaded")
        sheets = list(sheet_to_df_map.keys())
        dropdown_options = [{'label': _, 'value': _} for _ in sheets]
        out = [[
            html.Table(children=[
                html.Tr(children=[
                    html.Td(children=[
                        html.Div("Portfolio sheet:",
                                 id="portfolio_sheetname",
                                 style={"min-width": "120px"})
                    ]),
                    html.Td(children=[
                        dcc.Dropdown(
                            options=dropdown_options,
                            value=sheets[0],
                            multi=False,
                            clearable=False,
                            style={"padding-left": '2px'},
                            id="portfolio_sheetname_widget"
                        )
                    ], style={"min-width": "150px"})
                ])])
        ],
            [
            html.Table(children=[
                html.Tr(children=[
                        html.Td(children=[
                            html.Div("Covariance sheet:",
                                     id="covariance_sheetname",
                                     style={"min-width": "120px"})
                        ]),
                        html.Td(children=[
                            dcc.Dropdown(
                                options=dropdown_options,
                                value=sheets[1],
                                multi=False,
                                clearable=False,
                                style={"padding-left": '2px'},
                                id="covariance_sheetname_widget"
                            )
                        ], style={"min-width": "150px"})
                        ])])
        ],
            [
            html.Table(children=[
                html.Tr(children=[
                    html.Td(children=[
                        html.Div("Views sheet:",
                                 id="views_sheetname",
                                 style={"min-width": "120px"})
                    ]),
                    html.Td(children=[
                        dcc.Dropdown(
                            options=dropdown_options,
                            value=sheets[2],
                            multi=False,
                            clearable=False,
                            style={"padding-left": '2px'},
                            id="views_sheetname_widget"
                        )
                    ], style={"min-width": "150px"})
                ])])
        ]]
        logging.info("Sheet info gathered. Returning to UI...")
        packed_data = {key: df.to_csv() for key, df in zip(
            sheet_to_df_map, sheet_to_df_map.values())}
        return out[0], out[1], out[2], packed_data

    else:
        raise PreventUpdate


# @dash_app.callback(
#     [Output(component_id="portfolio-sheet", component_property="children"),
#      Output(component_id="views-sheet", component_property="children")],
#     [Input(component_id="portfolio_sheetname_widget", component_property="value"),
#      Input(component_id="covariance_sheetname_widget",
#            component_property="value"),
#      Input(component_id="views_sheetname_widget", component_property="value")],
#     [State(component_id="raw_input_dict", component_property="data")]
# )
# def load_keys(ptf_sheetname, cov_sheetname, views_sheetname, data):

# # Unpack the cache
#     json/
#     list_sheets = [ptf_sheetname, cov_sheetname, views_sheetname]
#     if len(set(list_sheets)) == len(list_sheets):
#         ptf_row_names = [
#             html.Td(children=[
#                 html.Div("Views sheet:",
#                          id="views_sheetname",
#                          style={"min-width": "120px"})
#             ]),
#             html.Td(children=[
#                 dcc.Dropdown(
#                     options=dropdown_options,
#                     value=sheets[2],
#                     multi=False,
#                     clearable=False,
#                     style={"padding-left": '2px'},
#                     id="views_sheetname_widget"
#                 )
#             ], style={"min-width": "150px"})
#         ]
#         views_col_names = [
#             html.Td(children=[
#                     html.Div("return row:",
#                              id="views_sheetname",
#                              style={"min-width": "120px"})
#                     ]),
#             html.Td(children=[
#                     dcc.Dropdown(
#                         options=dropdown_options,
#                         value=sheets[2],
#                         multi=False,
#                         clearable=False,
#                         style={"padding-left": '2px'},
#                         id="views_sheetname_widget"
#                     )
#                     ], style={"min-width": "150px"})
#         ]

#     else:
#         raise PreventUpdate


# def load_excel(data, filename):
#     # if prevent_updates == "True":
#     #     raise PreventUpdate
#     if data and filename:
#         logging.info("Loading custom portfolio...")
#         content_type, content_string = data.split(',')
#         decoded = base64.b64decode(content_string)
#         extension = filename.split('.')[-1]
#         if extension in ["xlsx", "xls"]:
#             sheet_to_df_map = pd.read_excel(
#                 io.BytesIO(decoded), sheet_name=None)
#         else:
#             raise Exception("Unsupported file type"
#                             "Please use .xls(x) file."
#                             "Found: {}".format(filename))
#         logging.info("Custom excel file loaded")
#         sheets = list(sheet_to_df_map.keys())
#         dropdown_options = [{'label': _, 'value': _} for _ in sheets]
#         out = [
#             html.Div("File {} successfully uploaded".format(filename)),
#             html.Table(children=[
#                 html.Tr(children=[
#                     html.Td(children=[
#                         html.Div("Portfolio sheet name:",
#                                  id="portfolio_sheetname")
#                     ]),
#                     html.Td(children=[
#                         dcc.Dropdown(
#                             options=dropdown_options,
#                             value=sheets[0],
#                             multi=False,
#                             clearable=False,
#                             style={"padding-left": '2px'},
#                             id="portfolio_sheetname_widget"
#                         )
#                     ], style={"min-width": "250px"})
#                 ]),
#                 html.Tr(children=[
#                     html.Td(children=[
#                         html.Div("Covariance sheet name:",
#                                  id="covariance_sheetname")
#                     ]),
#                     html.Td(children=[
#                         dcc.Dropdown(
#                             options=dropdown_options,
#                             value=sheets[1],
#                             multi=False,
#                             clearable=False,
#                             style={"padding-left": '2px'},
#                             id="covariance_sheetname_widget"
#                         )
#                     ], style={"min-width": "250px"})
#                 ]),
#                 html.Tr(children=[
#                     html.Td(children=[
#                         html.Div("Views sheet name:",
#                                  id="views_sheetname")
#                     ]),
#                     html.Td(children=[
#                         dcc.Dropdown(
#                             options=dropdown_options,
#                             value=sheets[2],
#                             multi=False,
#                             clearable=False,
#                             style={"padding-left": '2px'},
#                             id="covariance_sheetname_widget"
#                         )
#                     ], style={"min-width": "250px"})
#                 ]),
#             ])
#         ]
#         logging.info("Sheet info gathered. Returning to UI...")
#         return out
#     else:
#         raise PreventUpdate
if __name__ == "__main__":
    dash_app.run_server(debug=True)
