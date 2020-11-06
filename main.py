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


# # uporządkuj plik i wsadź to do klasy
# TRAININGS = 0
# SAMPLES_XY_1 = [[], []]
# SAMPLES_XY_2 = [[], []]


# def makeTraining(neu, samples1, samples2):

#     li1 = list(zip(samples1[0], samples1[1]))

#     for i in range(200):
#         for index, trainingTouple in enumerate(li1):
#             # print(index, trainingTouple)
#             neu.train(trainingTouple, 0)
#             neu.updateWeights()

#     li2 = list(zip(samples2[0], samples2[1]))

#     for i in range(200):
#         for index, trainingTouple in enumerate(li2):
#             # print(index, trainingTouple)
#             neu.train(trainingTouple, 1)
#             neu.updateWeights()
#     return neu


# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1('Hello Dash!'),
#     html.Div(id='output-div-1', children=[]),
#     html.Div(id='hidden2-div', children=[]),
#     html.P(id='x1'),

#     html.Div(style={'display': 'flex'}, children=[
#         dcc.Graph(id='activation-function',
#                   style={'width': '800px', 'height': '800px'}),

#         html.Div(children=[
#             html.Button('Training', id='btn-training', n_clicks=0, style={
#                 **styles.button}),

#             html.Button('Reset', id='btn-reset', n_clicks=0, style={
#                 **styles.button}),

#             html.Button('New samples', id='btn-new-samples', n_clicks=0, style={
#                 **styles.button}),

#             dcc.Input(id='input-trainings', type='number')
#         ])

#     ])
# ])

# # output przedstawia inputy przy tworzeniu sampli


# @app.callback(Output('hidden2-div', 'children'), Input('btn-new-samples', 'n_clicks'))
# def createNewSamples(n_clicks):
#     c1 = c.Classifier(2, 20)
#     x1, y1 = c1.getAllSamples()

#     c2 = c.Classifier(2, 20)
#     x2, y2 = c2.getAllSamples()

#     SAMPLES_XY_1 = [x1, y1]
#     SAMPLES_XY_2 = [x2, y2]

#     return [html.P(('New samples created times {}').format(n_clicks))]


# @app.callback(Output('activation-function', 'figure'), [Input('btn-training', 'n_clicks'), Input('hidden2-div', 'children')])
# def buttonClicked(n_clicks, childree):
#     fig = px.line(x=([-.5, -.25, 0, .25, .5]), y=(np.dot([-.5, -.25, 0, .25, .5], n_clicks)),
#                   labels={'x': 'xd', 'y': 'dxx'})

#     x = np.arange(0, 1.01, .01)
#     y = x.copy()

#     n = neuron.Neuron([1, 0.5], aft.LogisticFunction)

#     xx = x
#     yy = y

#     zz = []

#     n = makeTraining(n, SAMPLES_XY_1, SAMPLES_XY_2)

#     for _y in yy:
#         _z = []
#         for _x in xx:
#             _z.append(n.examine([_x, _y]))
#         zz.append(_z)

#     # n.examine([1, 6])

#     scatter = go.Scattergl(
#         x=[.25, .25],
#         y=[.4, .6],
#         mode='markers',
#         name='Sampled Data',
#     )

#     contour = None

#     if n_clicks > 0:
#         contour = go.Contour(
#             z=zz,
#             x=xx,
#             y=yy
#         )

#     fig = go.Figure(data=[scatter])

#     if contour != None:
#         fig = go.Figure(data=[contour, scatter])

#     fig.update_xaxes(range=[-.1, 1.1])
#     fig.update_yaxes(range=[-.1, 1.1])

#     return fig


# @app.callback(Output('btn-training', 'n_clicks'), Input('btn-reset', 'n_clicks'))
# def resetTraining(n_click):
#     return 0


# @app.callback(Output('output-div-1', 'children'), Input('input-trainings', 'value'))
# def setTrainings(value):
#     TRAININGS = value
#     return [html.P(('Traninings are set to {}').format(TRAININGS))]


# app.run_server(debug=True)

# ----------------------

neu = neuron.Neuron([1, 0.5], aft.LogisticFunction)

c1 = c.Classifier(1, 400)
x1, y1 = c1.getAllSamples()

li1 = list(zip(x1, y1))

for i in range(400):
    for index, trainingTouple in enumerate(li1):
        # print(index, trainingTouple)
        neu.train(trainingTouple, 0)
        neu.updateWeights()

c2 = c.Classifier(1, 400)
x2, y2 = c2.getAllSamples()

li2 = list(zip(x2, y2))
for i in range(400):
    for index, trainingTouple in enumerate(li2):
        # print(index, trainingTouple)
        neu.train(trainingTouple, 1)
        neu.updateWeights()

print()

print('li1:', li1[0])
print('expect 0:', neu.examine(li1[0]))
print('li2:', li2[0])
print('expect 1:', neu.examine(li2[0]))
