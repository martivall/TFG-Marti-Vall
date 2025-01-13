import plotly.express as px
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage
from skimage import io
import dash_bootstrap_components as dbc


app = Dash()

segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader("Viewer"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading",
                                type="circle",
                                children=[ html.Div(id='output-image-upload')
                                    # Graph
                                    
                                    #dcc.Graph(
                                       # id="graph",
                                       # figure=make_default_figure(),
                                       # config={
                                       #     "modeBarButtonsToAdd": [
                                       #         "drawrect",
                                       #         "drawopenpath",
                                       #         "eraseshape",
                                       #     ]
                                       # },
                                    #),
                                ],
                            )
                        ],
                    ),
                ]
            ),
            dbc.CardFooter(
                [
                    # Download links
                    html.A(id="download", download="classifier.json",),
                    html.Div(
                        children=[
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Download classified image",
                                        id="download-image-button",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        "Download classifier",
                                        id="download-button",
                                        #outline=True,
                                    ),
                                ],
                                size="lg",
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.A(id="download-image", download="classified-image.png",),
                ]
            ),
        ],

    )
]

app.layout = html.Div([
   
    dbc.Container(
            [
                dbc.Row( 
                    dcc.Upload(
                        id='upload-image', 
                        children=dbc.Button(
                            'Upload File', 
                            color="primary",
                            outline=True,
                            size="lg", 
                            style={"width": "100%"}),
                        multiple=True
                    ), 
                ),
                dbc.Row(
                    id="app-content",
                    children=[dbc.Col(segmentation, md=8)]
                    #children=[dbc.Col(segmentation, md=8), dbc.Col(sidebar, md=4)],
                ),
                #dbc.Row(dbc.Col(meta)),
            ],
            fluid=True,
        ),
    #html.Div(id='output-image-upload')
    #dcc.Graph(figure=fig, config=config)
])

def parse_contents(contents, filename, date):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    fig = px.imshow(pix)
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False, scaleanchor="x", showticklabels=False, zeroline=False
    )
    fig.update_layout(dragmode="drawopenpath")
    config = {
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ]
    }
    return html.Div([dcc.Graph(figure=fig, config=config)])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
    

if __name__ == '__main__':
    app.run(debug=True)