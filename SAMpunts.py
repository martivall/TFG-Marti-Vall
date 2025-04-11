import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import dash
from dash import ALL
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage
import dash_daq as daq  # Importar Dash DAQ
from skimage import io
from PIL import Image
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
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col(
                                dcc.Upload(
                                    id='upload-image',
                                    children=dbc.Button('Upload File', color="primary", outline=True),
                                ),
                                width="auto"
                            ),
                            dbc.Col(width=True),
                            dbc.Col([
                                dbc.Button('Save Current Segmentation', id='save-segmentation-button', color="primary", className="me-2", outline=True),
                                dbc.Button('Overwrite Image', id='overwrite-image-button', color="primary", outline=True),
                            ], width="auto"),
                        ], align="center")
                    ),

                    dbc.CardBody([
                        dcc.Store(id='stored-figure', data={}),  # Almacena la figura procesada
                        dcc.Store(id='loading-flag', data=False),  # Estado de carga
                        dcc.Store(id='points-data', data={'x': [], 'y': [], 'color': []}),  # Almacena puntos
                        dcc.Store(id='button-state', data={"positive": False, "negative": False}),
                        dcc.Store(id='mask-bitmap', data=None), 
                        dcc.Store(id='segmentation-gallery', data=[]),
                        dcc.Store(id='selected-mask', data=None),
                        dcc.Store(id='alternative-masks', data=[]),

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

                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Download image", id="download-image-button", color="secondary", className="me-2", style={"whiteSpace": "nowrap"}),
                                dbc.Button("Download mask bitmap", id="download-mask-button", color="secondary", style={"whiteSpace": "nowrap"}),    
                            ], width="auto"),
                            dbc.Col(width=True),
                            dbc.Col(
                                dbc.Button("Try segmentation", id="segment-button", color="primary", style={"whiteSpace": "nowrap"}),
                                width="auto"
                            )                         
                        ], align="center"),
                        

                        dcc.Download(id="download-mask"), 

                        dcc.Download(id="download-image"),
                    ]),
                ]),
                dbc.Card([
                    dbc.CardHeader("Alternative Masks"),
                    dbc.CardBody(id="alternative-masks-thumbnails")
                ], className="mt-3"),
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Tools"),
                    dbc.CardFooter([
                        dbc.Row([
                            dbc.Col(
                                dbc.ButtonGroup([
                                    dbc.Button("Punt Positiu", id="btn-positive", color="success", outline=True),
                                    dbc.Button("Punt Negatiu", id="btn-negative", color="danger", outline=True),
                            
                                ], style={"width": "100%"}
                                ),
                                width="auto"
                            ),
                            dbc.Col(width=True),
                            dbc.Col(
                                dbc.Button("Clean All", id="btn-clean", color="secondary", outline=True),
                                width="auto"
                            ),
                        ]),

                        dcc.Checklist(
                            id='checklist',
                            options=['Recàlcul automàtic']
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
                ]),
                dbc.Card([
                    dbc.CardHeader("Segmentations Gallery"),
                    dbc.CardBody(id="segmentations-thumbnails")
                    ])
            ], md=4),
                    
        ]),
        dbc.Modal([
            dbc.ModalHeader("Enlarged segmentation"),
            dbc.ModalBody(html.Img(id="enlarged-segmentation", style={"width": "100%"})),
            dbc.ModalFooter(
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Download", id="download-selected-mask-button", color="primary", className="me-2"),
                        width="auto"
                    )
                ])
            ),
        ],
        id="modal-segmentation",
        size="lg",
        is_open=False,
        ),

        dcc.Download(id="download-selected-mask"),

    ], fluid=True)
])

# Funciones para procesar imágenes
def parse_contents(contents):
    img = image_string_to_PILImage(contents).convert("RGB")
    pix = np.array(img)
    fig = px.imshow(pix)
    fig.update_xaxes(showgrid=False, ticks='', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks='', showticklabels=False, zeroline=False)
    fig.update_layout(template=None, dragmode="select")
    return fig

def parse_segmentation(contents):
    img = image_string_to_PILImage(contents).convert("RGB")
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

def parse_box_segmentation(contents, box, input_points, input_labels, return_all_masks=False):
    img = image_string_to_PILImage(contents).convert("RGB")
    pix = np.array(img)
    image_rgb = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box if box.size > 0 else None,
        point_coords=input_points if input_points.size > 0 else None,
        point_labels=input_labels if input_labels.size > 0 else None,
        multimask_output=True
    )

    if return_all_masks:
        alternative_thumbnails = []
        for i, m in enumerate(masks):
            if m.ndim == 3:
                mask = m[0]
            else:
                mask = m

            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            alternative_thumbnails.append({
                "src": f"data:image/png;base64,{base64_img}",
                "score": float(scores[i])
            })
        
        alternative_thumbnails.sort(key=lambda x: x['score'], reverse=True)

        return masks, scores, alternative_thumbnails

    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
    detections = detections[detections.area == np.max(detections.area)]
    mask = detections.mask[0] 
    segmented_image = mask_annotator.annotate(scene=pix.copy(), detections=detections)

    fig = px.imshow(segmented_image)
    fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False)
    fig.update_layout(template=None, dragmode="select")

    return fig, mask


@app.callback(
    Output('stored-figure', 'data'),
    Output('loading-flag', 'data'),
    Output('mask-bitmap', 'data'),
    Output('alternative-masks', 'data'),
    Input('segment-button', 'n_clicks'),
    Input('upload-image', 'contents'),
    State('output-image-upload', 'relayoutData'),
    Input('points-data','data'),
    State('checklist', 'value')
)
def update_stored_figure(nclicks, content, relayout_data, points_data, check):
    if content is None:
        return {}, False, None, []

    if ctx.triggered_id == "segment-button":
        loading_flag = True
        if "shapes" in relayout_data:
            last_shape = relayout_data["shapes"][-1]
            x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
            x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
            box = np.array([x0, y0, x1, y1])

            input_points = np.array([[int(x), int(y)] for x, y in zip(points_data["x"], points_data["y"])]).reshape(-1, 2)
            input_labels = np.array([int(c) for c in points_data["color"]])

            masks, scores, alternative_thumbnails = parse_box_segmentation(
                content, box, input_points, input_labels, return_all_masks=True
            )

            selected_mask = masks[np.argmax(scores)]

            # Generar figura con la mejor máscara
            mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
            detections = detections[detections.area == np.max(detections.area)]
            segmented_image = mask_annotator.annotate(scene=np.array(image_string_to_PILImage(content)), detections=detections)

            fig = px.imshow(segmented_image)
            fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False)
            fig.update_layout(template=None, dragmode="select")
        else:
            fig = parse_segmentation(content)
            selected_mask = None
            alternative_thumbnails = []
    elif ctx.triggered_id == "points-data":
        if check == ['Recàlcul automàtic']:
            if not points_data["x"]:
                fig = parse_contents(content)
                return fig.to_dict(), False, None, []
            else:
                box = np.array([])
                input_points = np.array([[int(x), int(y)] for x, y in zip(points_data["x"], points_data["y"])]).reshape(-1, 2)
                input_labels = np.array([int(c) for c in points_data["color"]])
                masks, scores, alternative_thumbnails = parse_box_segmentation(
                    content, box, input_points, input_labels, return_all_masks=True
                )
                selected_mask = masks[np.argmax(scores)]

                mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
                detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
                detections = detections[detections.area == np.max(detections.area)]
                segmented_image = mask_annotator.annotate(scene=np.array(image_string_to_PILImage(content)), detections=detections)

                fig = px.imshow(segmented_image)
                fig.update_xaxes(showgrid=False, ticks= '', showticklabels=False, zeroline=False)
                fig.update_yaxes(showgrid=False, scaleanchor="x", ticks= '', showticklabels=False, zeroline=False)
                fig.update_layout(template=None, dragmode="select")
        else:
            fig = parse_contents(content)
            selected_mask = None
            alternative_thumbnails = []
    elif ctx.triggered_id == "upload-image":
        fig = parse_contents(content)
        selected_mask = None
        alternative_thumbnails = []
    else:
        return {}, False, None, []

    return fig.to_dict(), False, selected_mask.tolist() if selected_mask is not None else None, alternative_thumbnails


@app.callback(
    Output('alternative-masks-thumbnails', 'children'),
    Input('alternative-masks', 'data')
)
def update_alt_thumbnails(gallery):
    if not gallery:
        return html.P("No alternative masks.")

    return html.Div([
        html.Div([
            html.Img(
                src=item["src"],
                id={'type': 'alt-thumbnail', 'index': i},
                n_clicks=0,
                style={"height": "100px", "border": "1px solid #ccc", "cursor": "pointer", "display": "block", "marginBottom": "5px"}
            ),
            html.Small(f"Score: {item['score']:.3f}", style={"display": "block", "textAlign": "center"})
        ],
        style={"margin": "5px", "textAlign": "center"})
        for i, item in enumerate(gallery)
    ], style={"display": "flex", "flexWrap": "wrap"})


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
    Input('btn-clean', 'n_clicks'),  # Nuevo botón de limpieza
    State('button-state', 'data'),
    State('points-data', 'data'),
    prevent_initial_call=True
)
def update_points(clickData, clean_click, button_state, points_data):
    triggered_id = ctx.triggered_id  # ctx = dash.callback_context en versiones nuevas

    if triggered_id == 'btn-clean':
        return {"x": [], "y": [], "color": []}

    if triggered_id == 'output-image-upload' and clickData:
        x_click = clickData["points"][0]["x"]
        y_click = clickData["points"][0]["y"]

        if button_state["positive"]:
            color_value = 1
        elif button_state["negative"]:
            color_value = 0
        else:
            return dash.no_update  # Ningún botón activo

        points_data["x"].append(x_click)
        points_data["y"].append(y_click)
        points_data["color"].append(color_value)

        return points_data

    return dash.no_update


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


# Callback para guardar segmentaciones en la galería
@app.callback(
    Output('segmentation-gallery', 'data'),
    Input('save-segmentation-button', 'n_clicks'),
    State('mask-bitmap', 'data'),
    State('segmentation-gallery', 'data'),
    prevent_initial_call=True
)
def save_current_segmentation(n_clicks, mask_data, gallery):
    if mask_data is None:
        return gallery

    mask_array = np.array(mask_data, dtype=np.uint8) * 255
    img = Image.fromarray(mask_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    gallery.append(f"data:image/png;base64,{base64_img}")
    return gallery


# Callback para mostrar los thumbnails
@app.callback(
    Output('segmentations-thumbnails', 'children'),
    Input('segmentation-gallery', 'data')
)
def update_thumbnail_gallery(gallery):
    if not gallery:
        return html.P("No saved segmentations yet.")
    
    return html.Div([
        html.Img(
            src=img_src,
            id={'type': 'thumbnail', 'index': i},
            n_clicks=0,
            style={"height": "100px", "margin": "5px", "border": "1px solid #ccc", "cursor": "pointer"}
        )
        for i, img_src in enumerate(gallery)
    ], style={"display": "flex", "flexWrap": "wrap"})


# Callback para abrir el modal con la imagen clicada
@app.callback(
    Output("modal-segmentation", "is_open"),
    Output("enlarged-segmentation", "src"),
    Output("selected-mask", "data"),
    Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks'),
    Input({'type': 'alt-thumbnail', 'index': ALL}, 'n_clicks'),
    State('segmentation-gallery', 'data'),
    State('alternative-masks', 'data'),
    State("modal-segmentation", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(thumbnail_clicks, alt_clicks, gallery, alt_gallery, is_open):
    ctx_id = ctx.triggered_id

    if isinstance(ctx_id, dict):
        idx = ctx_id.get("index")
        if ctx_id.get("type") == "thumbnail" and 0 <= idx < len(gallery):
            if thumbnail_clicks[idx] > 0:
                return True, gallery[idx], gallery[idx]
        elif ctx_id.get("type") == "alt-thumbnail" and 0 <= idx < len(alt_gallery):
            if alt_clicks[idx] > 0:
                return True, alt_gallery[idx]["src"], alt_gallery[idx]["src"]

    return is_open, dash.no_update, dash.no_update



# Callback para la descarga del modal
@app.callback(
    Output("download-selected-mask", "data"),
    Input("download-selected-mask-button", "n_clicks"),
    State("selected-mask", "data"),
    prevent_initial_call=True
)
def download_selected_mask(n_clicks, mask_src):
    if not mask_src:
        return dash.no_update

    # Quitar el encabezado base64
    header, base64_data = mask_src.split(",", 1)
    img_bytes = base64.b64decode(base64_data)

    return dcc.send_bytes(img_bytes, filename="selected_mask.png")


# Callback para descargar la máscara
@app.callback(
    Output('download-mask', 'data'),
    Input('download-mask-button', 'n_clicks'),
    State('mask-bitmap', 'data'),
    prevent_initial_call=True
)
def download_mask(n_clicks, mask_data):
    if mask_data is None:
        return dash.no_update

    mask_array = np.array(mask_data, dtype=np.uint8) * 255
    img = Image.fromarray(mask_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return dcc.send_bytes(buf.read(), filename="mask_bitmap.png")


# Callback para descargar la imagen
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
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  # Quitar todos los márgenes
        paper_bgcolor='rgba(0,0,0,0)',    # Fondo completamente transparente
        plot_bgcolor='rgba(0,0,0,0)',     # Fondo del gráfico transparente
    )

    img_bytes = pio.to_image(fig, format="png", width=fig.layout.width, height=fig.layout.height)

    #img_bytes = pio.to_image(fig, format="png")

    return dcc.send_bytes(img_bytes, filename="imagen.png")


# Callback para actualizar la imagen con los trazados
@app.callback(
    Output('upload-image', 'contents'),
    Input('overwrite-image-button', 'n_clicks'),
    State('upload-image', 'contents'),  # <- usamos esta como base
    State('output-image-upload', 'relayoutData'),
    prevent_initial_call=True
)
def overwrite_image(n_clicks, original_contents, relayout_data):
    if not original_contents:
        return dash.no_update

    # Cargar imagen original, sin segmentaciones
    img = image_string_to_PILImage(original_contents).convert("RGB")
    pix = np.array(img)

    # Crear figura base desde la imagen original
    fig = px.imshow(pix)
    fig.update_xaxes(showgrid=False, ticks='', showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, scaleanchor="x", ticks='', showticklabels=False, zeroline=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        dragmode="select"
    )

    # Agregar SOLO los trazados dibujados por el usuario
    if relayout_data and "shapes" in relayout_data:
        for shape in relayout_data["shapes"]:
            if shape["type"] == "line":
                fig.add_shape(
                    type="line",
                    x0=shape["x0"], y0=shape["y0"],
                    x1=shape["x1"], y1=shape["y1"],
                    line=dict(color=shape["line"]["color"], width=shape["line"]["width"])
                )
            elif shape["type"] == "path":
                fig.add_shape(
                    type="path",
                    path=shape["path"],
                    line=dict(color=shape["line"]["color"], width=shape["line"]["width"])
                )

    # Exportar figura como imagen (png)
    img_bytes = pio.to_image(
    fig,
    format="png",
    width=pix.shape[1],  # Ancho original
    height=pix.shape[0]  # Alto original
    )
    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
