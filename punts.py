import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage
from skimage import io
import dash_bootstrap_components as dbc
import torch
import cv2
import supervision as sv
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

# Configuración del modelo SAM
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_b_01ec64.pth')
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

# Inicializar la aplicación Dash
external_stylesheets = [dbc.themes.BOOTSTRAP, "styles.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout de la aplicación
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(html.H1("Image segmentation tool based on SAM"))))
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(dcc.Upload(
                        id='upload-image',
                        children=dbc.Button('Upload File', color="primary", outline=True),
                    )),
                    dbc.CardBody([
                        dcc.Store(id='stored-figure', data={}),  # Almacena la figura procesada
                        dcc.Store(id='loading-flag', data=False),  # Estado de carga
                        dcc.Store(id='points-data', data={'x': [], 'y': [], 'color': []}),  # Almacena puntos

                        dcc.Loading(
                            id="segmentations-loading",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='output-image-upload',
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
                                ),
                            ],
                        ),
                    ]),
                    dbc.CardFooter([
                        #dbc.ButtonGroup([
                        #   dbc.Button("Punto Positivo", id="btn-positive", color="success", outline=True),
                        #   dbc.Button("Punto Negativo", id="btn-negative", color="danger", outline=True),
                        #], size="lg", style={"width": "100%"}),

                        dbc.ButtonGroup([
                            dbc.Button("Download classified image", id="download-image-button", outline=True),
                            dbc.Button("Try segmentation", id="segment-button"),
                        ], size="lg", style={"width": "100%"}),

                        html.A(id="download-image", download="classified-image.png"),
                    ]),
                ])
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tools"),
                    dbc.CardFooter([
                        dbc.ButtonGroup([
                            dbc.Button("Punto Positivo", id="btn-positive", color="success", outline=True),
                            dbc.Button("Punto Negativo", id="btn-negative", color="danger", outline=True),
                        ], size="lg", style={"width": "100%"})
                    ])
                ])
            ], md=4),
                    
        ]),
    ], fluid=True)
])

# Funciones para procesar imágenes
def parse_contents(contents):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    fig = px.imshow(pix)
    fig.update_xaxes(showgrid=False, ticks='', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks='', showticklabels=False, zeroline=False)
    fig.update_layout(template=None, dragmode="select")
    return fig

def parse_segmentation(contents):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    image_rgb = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(pix.copy(), detections)
    fig = px.imshow(annotated_image)
    fig.update_xaxes(showgrid=False, ticks='', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks='', showticklabels=False, zeroline=False)
    fig.update_layout(template=None, dragmode="select")
    return fig

@app.callback(
    Output('stored-figure', 'data'),
    Output('loading-flag', 'data'),
    Input('segment-button', 'n_clicks'),
    Input('upload-image', 'contents'),
    Input('output-image-upload', 'relayoutData')
)
def update_stored_figure(nclicks, content, relayout_data):
    if content is None:
        return {}, False

    if ctx.triggered_id == "segment-button":
        loading_flag = True
        fig = parse_segmentation(content)
    elif ctx.triggered_id == "upload-image":
        loading_flag = True
        fig = parse_contents(content)
    else:
        return {}, False

    return fig.to_dict(), False

# Callback para añadir puntos positivos y negativos
@app.callback(
    Output('points-data', 'data'),
    Input('output-image-upload', 'clickData'),
    State('btn-positive', 'n_clicks'),
    State('btn-negative', 'n_clicks'),
    State('points-data', 'data'),
    prevent_initial_call=True
)
def add_point(clickData, n_pos, n_neg, points_data):
    if not clickData:
        return dash.no_update

    x_click = clickData["points"][0]["x"]
    y_click = clickData["points"][0]["y"]
    color = "green" if (n_pos or 0) > (n_neg or 0) else "red"

    points_data["x"].append(x_click)
    points_data["y"].append(y_click)
    points_data["color"].append(color)

    return points_data

# Callback para actualizar la imagen con los puntos
@app.callback(
    Output('output-image-upload', 'figure'),
    Input('stored-figure', 'data'),
    Input('points-data', 'data'),
    State('loading-flag', 'data')
)
def update_graph(stored_data, points_data, is_loading):
    if not stored_data or is_loading:
        return dash.no_update

    fig = go.Figure(stored_data)

    if points_data["x"]:
        fig.add_trace(go.Scatter(
            x=points_data["x"], y=points_data["y"],
            mode="markers",
            marker=dict(size=10, color=points_data["color"], symbol="circle"),
            name="Puntos"
        ))

    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
