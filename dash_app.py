import plotly.express as px
from dash import Dash, dcc, html
from skimage import io

img = io.imread('perro.jpg')
fig = px.imshow(img)
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


app = Dash()

app.layout = html.Div([
    dcc.Upload(html.Button('Upload File')),
    html.H3("Drag and draw annotations"),
    dcc.Graph(figure=fig, config=config)
])

if __name__ == '__main__':
    app.run(debug=True)