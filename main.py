import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

import styles
import neuron
from neuron import ActivationFunctionTypes as aft
import classifier as c


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Hello Dash!'),

    html.Button('Button 2', id='btn', n_clicks=0, style=styles.button),

    dcc.Graph(id='activation-function')
])


@app.callback(Output('activation-function', 'figure'), Input('btn', 'n_clicks'))
def buttonClicked(button):
    fig = px.line(x=([-.5, -.25, 0, .25, .5]), y=(np.dot([-.5, -.25, 0, .25, .5], button)),
                  labels={'x': 'xd', 'y': 'dxx'})

    x = np.arange(0, 1.01, .01)
    y = x.copy()

    n = neuron.Neuron([1, 0.5], aft.LogisticFunction)

    xx = x
    yy = y

    zz = []

    for _y in yy:
        _z = []
        for _x in xx:
            _z.append(n.examine([_x, _y]))
        zz.append(_z)

    n.examine([1, 6])

    fig = go.Figure(data=go.Contour(
        z=zz,
        x=xx,
        y=yy
    ))

    print('jebadło')

    fig.update_xaxes(range=[-.1, 1.1])
    fig.update_yaxes(range=[-.1, 1.1])

    return fig


app.run_server(debug=True)


# neu = neuron.Neuron([1, 0.5], aft.HeaviSideStepFunction)

# c1 = c.Classifier(2, 400)
# x1, y1 = c1.getAllSamples()

# li1 = list(zip(x1, y1))

# for i in range(200):
#     for index, trainingTouple in enumerate(li1):
#         # print(index, trainingTouple)
#         neu.train(trainingTouple, 0)
#         neu.updateWeights()


# c2 = c.Classifier(2, 400)
# x2, y2 = c2.getAllSamples()

# li2 = list(zip(x2, y2))
# for i in range(200):
#     for index, trainingTouple in enumerate(li2):
#         # print(index, trainingTouple)
#         neu.train(trainingTouple, 1)
#         neu.updateWeights()
