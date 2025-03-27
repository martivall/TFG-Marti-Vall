import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage
import dash_daq as daq  # Importar Dash DAQ
from skimage import io
import dash_bootstrap_components as dbc
import base64
import io
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
                        dcc.Store(id='button-state', data={"positive": False, "negative": False}),

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
                            dbc.Button("Download classified image", id="download-image-button", color="secondary"),
                            dbc.Button("Try segmentation", id="segment-button"),
                        ], size="lg", style={"width": "100%"}),

                         dcc.Download(id="download-image"),
                    ]),
                ])
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tools"),
                    dbc.CardFooter([
                        dbc.ButtonGroup([
                            dbc.Button("Punt Positiu", id="btn-positive", color="success", outline=True),
                            dbc.Button("Punt Negatiu", id="btn-negative", color="danger", outline=True),
                        ], size="lg", style={"width": "100%"}
                        ),

                        daq.ColorPicker(
                            id='color-picker',
                            label='Color del traçat',
                            value={'hex': '#FF0000'}  # Valor por defecto: Rojo
                        ),
                        
                        dcc.Slider(
                            id='line-width-slider',
                            min=1, max=10, step=1,
                            value=2,  # Valor por defecto
                            marks={i: str(i) for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        )
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

def parse_box_segmentation(contents, box, input_points, input_labels):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    image_rgb = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    if input_points.size == 0:
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)
    else:
        # print(input_points)
        # print(input_labels)
        # print(f"Forma de input_points: {input_points.shape}")
        # print(f"Forma de input_labels: {input_labels.shape}")
        masks, scores, logits = mask_predictor.predict(point_coords=input_points, point_labels=input_labels, box=box, multimask_output=False)
    box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks),mask=masks)
    detections = detections[detections.area == np.max(detections.area)]
    segmented_image = mask_annotator.annotate(scene=pix.copy(), detections=detections)
    segmented_image = box_annotator.annotate(scene=segmented_image.copy(), detections=detections)
    fig = px.imshow(segmented_image)
    fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False)
    fig.update_layout(template=None,dragmode="select")
    return fig

@app.callback(
    Output('stored-figure', 'data'),
    Output('loading-flag', 'data'),
    Input('segment-button', 'n_clicks'),
    Input('upload-image', 'contents'),
    State('output-image-upload', 'relayoutData'),
    State('points-data','data')
)
def update_stored_figure(nclicks, content, relayout_data, points_data):
    if content is None:
        return {}, False
    if ctx.triggered_id == "segment-button":
        loading_flag = True
        if "shapes" in relayout_data:
            last_shape = relayout_data["shapes"][-1]
            x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
            x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
            box = np.array([x0, y0, x1, y1])
            input_points = np.empty((0, 2), dtype=int)  # Array vacío con forma (N,2) para coordenadas
            input_labels = np.empty((0,), dtype=int)  # Array vacío con forma (N,) para etiquetas
            if points_data["x"]:
                for x, y, color in zip(points_data["x"], points_data["y"], points_data["color"]):
                    point = np.array([[int(x), int(y)]])  # Convertir x, y a enteros
                    label = np.array(int(color))  # Convertir color a entero (0 o 1)

                    input_points = np.vstack((input_points, point))  
                    input_labels = np.append(input_labels, label)  
            fig = parse_box_segmentation(content, box, input_points, input_labels)
        else:    
            fig = parse_segmentation(content)
    elif ctx.triggered_id == "upload-image":
        is_loading = True
        fig = parse_contents(content)
    else:
        return {}, False
    return fig.to_dict(), False

@app.callback(
    Output("button-state", "data"),
    Input("btn-positive", "n_clicks"),
    Input("btn-negative", "n_clicks"),
    State("button-state", "data"),
    prevent_initial_call=True
)
def toggle_buttons(n_clicks_pos, n_clicks_neg, state):
    ctx = dash.callback_context
    if not ctx.triggered_id:
        return state  # No cambiar si no hay interacción

    if ctx.triggered_id == "btn-positive":
        return {"positive": not state["positive"], "negative": False}  # Activa positivo y desactiva negativo
    elif ctx.triggered_id == "btn-negative":
        return {"positive": False, "negative": not state["negative"]}  # Activa negativo y desactiva positivo
    
    return state  # No cambiar si no se detecta interacción

@app.callback(
    Output("btn-positive", "active"),
    Output("btn-negative", "active"),
    Input("button-state", "data")
)
def update_button_states(button_state):
    return button_state["positive"], button_state["negative"]

# Callback para añadir puntos positivos y negativos
@app.callback(
    Output('points-data', 'data'),
    Input('output-image-upload', 'clickData'),
    State('button-state', 'data'),  # Obtenemos qué botón está activo
    State('points-data', 'data'),
    prevent_initial_call=True
)
def add_point(clickData, button_state, points_data):
    if not clickData:
        return dash.no_update  # No hacer nada si no hay clic

    x_click = clickData["points"][0]["x"]
    y_click = clickData["points"][0]["y"]

    # Determinar el valor de color según qué botón está activo
    if button_state["positive"]:
        color_value = 1  # Verde
    elif button_state["negative"]:
        color_value = 0  # Rojo
    else:
        return dash.no_update  # Si ningún botón está activo, no hacer nada

    # Agregar el punto con su color correspondiente
    points_data["x"].append(x_click)
    points_data["y"].append(y_click)
    points_data["color"].append(color_value)

    return points_data


# Callback para actualizar la imagen con los puntos
@app.callback(
    Output('output-image-upload', 'figure'),
    Input('stored-figure', 'data'),
    Input('points-data', 'data'),
    Input("color-picker", "value"),  # Leer el color directamente
    Input("line-width-slider", "value"),  # Leer el grosor directamente
    State('loading-flag', 'data')
)
def update_graph(stored_data, points_data, color_value, line_width, is_loading):
    if not stored_data or is_loading:
        return dash.no_update

    fig = go.Figure(stored_data)

    fig.update_layout(
        newshape=dict(
            line=dict(
                color=color_value["hex"],  # Color del trazo
                width=line_width  # Grosor del trazo
            )
        )
    )

    if points_data["x"]:
        # Convertir 0 → "red" y 1 → "green"
        colors = ["red" if c == 0 else "green" for c in points_data["color"]]

        fig.add_trace(go.Scatter(
            x=points_data["x"], y=points_data["y"],
            mode="markers",
            marker=dict(size=8, color=colors, symbol="circle"),
            name="Puntos"
        ))

    return fig

"""
@app.callback(
    Output("output-image-upload", "figure", allow_duplicate=True),
    State("stored-figure", "data"),
    Input("color-picker", "value"),  # Leer el color directamente
    Input("line-width-slider", "value"),  # Leer el grosor directamente
    prevent_initial_call=True
)
def update_draw_color(stored_data, color_value, line_width):
    print(color_value)
    if not stored_data:
        return dash.no_update
    print('stored data')
    fig = go.Figure(stored_data)

    # Configuración por defecto del dibujo de líneas
    fig.update_layout(
        newshape=dict(
            line=dict(
                color=color_value["hex"],  # Color del trazo
                width=line_width  # Grosor del trazo
            )
        )
    )

    return fig
"""

@app.callback(
    Output('download-image', 'data'),
    Input('download-image-button', 'n_clicks'),
    State('stored-figure', 'data'),
    State('output-image-upload', 'relayoutData'),  # Capturar dibujos
    prevent_initial_call=True
)
def save_image(n_clicks, stored_data, relayout_data):
    if not stored_data:
        return dash.no_update

    # Crear figura sin los puntos
    fig = go.Figure(stored_data)

    # Extraer formas desde relayoutData
    if relayout_data and "shapes" in relayout_data:
        for shape in relayout_data["shapes"]:
            shape_type = shape.get("type", "")

            # Para líneas rectas normales
            if shape_type == "line":
                fig.add_shape(
                    type="line",
                    x0=shape["x0"], y0=shape["y0"],
                    x1=shape["x1"], y1=shape["y1"],
                    line=dict(color=shape["line"]["color"], width=shape["line"]["width"])
                )

            # Para dibujos a mano alzada (Draw Open Freeform)
            elif shape_type == "path":
                fig.add_shape(
                    type="path",
                    path=shape["path"],  # La trayectoria dibujada
                    line=dict(color=shape["line"]["color"], width=shape["line"]["width"])
                )

    # Convertir a imagen (PNG)
    img_bytes = pio.to_image(fig, format="png")

    return dcc.send_bytes(img_bytes, filename="imagen_sin_puntos.png")

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
