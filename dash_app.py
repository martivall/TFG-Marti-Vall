import plotly.express as px
import numpy as np
from dash import Dash, dcc, html
from dash_canvas import DashCanvas
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url, image_string_to_PILImage
from skimage import io


app = Dash()

app.layout = html.Div([
    dcc.Upload(
        id='upload-image', 
        children=html.Button('Upload File'),
        multiple=True
    ),
    html.H3("Drag and draw annotations"),
    html.Div(id='output-image-upload')
    #dcc.Graph(figure=fig, config=config)
])

def parse_contents(contents, filename, date):
    img = image_string_to_PILImage(contents)
    pix = np.array(img)
    fig = px.imshow(pix)
    fig.update_layout(dragmode="drawrect")
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