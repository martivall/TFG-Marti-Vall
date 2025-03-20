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
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
import json

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_b_01ec64.pth')
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

external_stylesheets = [dbc.themes.BOOTSTRAP, "styles.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

header = [
    dbc.Card(
        dbc.CardBody(
            html.H1("Image segmentation tool based on SAM", className="card-title"),
        )
    )
]

segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader(dcc.Upload(
                id='upload-image',
                children=dbc.Button('Upload File', color="primary", outline=True),
            )),
            dbc.CardBody([
                dcc.Store(id='stored-figure', data={}),  # Almacena la figura procesada
                dcc.Store(id='loading-flag', data=False),  # Nuevo Store para estado de carga
                dcc.Loading(
                    id="segmentations-loading",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id='output-image-upload',
                            config={
                                'modeBarButtonsToRemove': ['zoom', 'pan'],
                                "modeBarButtonsToAdd": [
                                    "select", "drawline", "drawopenpath", "drawclosedpath",
                                    "drawcircle", "drawrect", "eraseshape"
                                ]
                            },
                        ),
                    ],
                )
            ]),
            dbc.CardFooter([
                html.A(id="download", download="classifier.json"),
                dbc.ButtonGroup([
                    dbc.Button("Download classified image", id="download-image-button", outline=True),
                    dbc.Button("Try segmentation", id="segment-button"),
                ], size="lg", style={"width": "100%"}),
                html.A(id="download-image", download="classified-image.png"),
            ]),
        ],
    )
]

sidebar = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Tools"),
        ],
    ),
]

app.layout = html.Div([
    dbc.Container([
        dbc.Row(header),
        dbc.Row(id="app-content", children=[dbc.Col(segmentation, md=8), dbc.Col(sidebar, md=4)]),
    ], fluid=True)
])

def parse_contents(contents):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    fig = px.imshow(pix)
    fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False
    )
    fig.update_layout(
        template=None,
        dragmode="select"
    )
    return fig

def parse_segmentation(contents):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    image_bgr = pix
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(image_bgr.copy(), detections)
    fig = px.imshow(annotated_image)
    fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False
    )
    fig.update_layout(
        template=None,
        dragmode="select"
    )
    return fig

def parse_box_segmentation(contents, box):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    image_bgr = pix
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    masks, scores, logits = mask_predictor.predict( box=box, multimask_output=True)
    box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    segmented_image = box_annotator.annotate(scene=segmented_image.copy(), detections=detections)
    fig = px.imshow(segmented_image)
    fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
    fig.update_yaxes(
        showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False
    )
    fig.update_layout(
        template=None,
        dragmode="select"
    )
    return fig

@app.callback(
    Output('stored-figure', 'data'),  # Guarda la figura procesada en dcc.Store
    Output('loading-flag', 'data'),  # Activa/Desactiva el estado de carga
    Input('segment-button', 'n_clicks'),
    Input('upload-image', 'contents'),
    Input('output-image-upload', 'relayoutData')
)
def update_stored_figure(nclicks, content, relayout_data):
    if content is None:
        return {}, False
    if ctx.triggered_id == "segment-button":
        loading_flag = True
        if "shapes" in relayout_data:
            last_shape = relayout_data["shapes"][-1]
            x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
            x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
            box = np.array([x0, y0, x1, y1])
            print(box)
            fig = parse_box_segmentation(content, box)
        else:    
            fig = parse_segmentation(content)
    elif ctx.triggered_id == "upload-image":
        is_loading = True
        fig = parse_contents(content)
    else:
        return {}, False
    return fig.to_dict(), False

@app.callback(
    Output('output-image-upload', 'figure'),
    Input('stored-figure', 'data'),
    State('loading-flag', 'data')
)
def update_graph(stored_data, is_loading):
    if not stored_data or is_loading:
        return dash.no_update
    return go.Figure(stored_data)

if __name__ == '__main__':
    app.run_server(debug=True)