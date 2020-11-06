import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import numpy as np
import plotly.graph_objects as go

import styles
import classifier as c
import neuron
from neuron import ActivationFunctionTypes as aft


class dashServer:
    defaultState = {
        'trainings': 0,
        'samples1': [[], []],
        'samples2': [[], []],
    }

    def __init__(self):
        self.state = self.defaultState.copy()

        self.app = dash.Dash(__name__)
        self.__setLayout()
        self.__configCallbacks()
        self.app.run_server(debug=True)

    def __setLayout(self):
        self.app.layout = html.Div(style={'display': 'flex'}, children=[
            dcc.Graph(id='contour-plot',
                      style={'width': '800px', 'height': '800px'}),

            html.Div(style={'height': 'fit-content', 'width': '500px', 'display': 'flex', 'justifyContent': 'space-between'}, children=[
                html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
                    html.Button('Training', id='btn-training', n_clicks=0, style={
                        **styles.button}),

                    dcc.Input(id='input-trainings', type='number'),
                ]),

                html.Button('Reset', id='btn-reset', n_clicks=0, style={
                    **styles.button}),

                html.Button('New samples', id='btn-new-samples', n_clicks=0, style={
                    **styles.button})
            ]),

            html.P(id='output-1'),
            html.P(id='output-2'),
        ])

    def __configCallbacks(self):
        @self.app.callback(Output('output-1', 'children'), Input('btn-new-samples', 'n_clicks'))
        def createNewSamples(n_clicks):
            c1 = c.Classifier(1, 20)
            x1, y1 = c1.getAllSamples()

            c2 = c.Classifier(1, 20)
            x2, y2 = c2.getAllSamples()

            self.state.update({'samples1': [x1, y1]})
            self.state.update({'samples2': [x2, y2]})

            return []

        @self.app.callback([Output('btn-training', 'n_clicks'), Output('btn-new-samples', 'n_clicks')], Input('btn-reset', 'n_clicks'))
        def resetTrainings(n_clicks):
            return 0, 0

        @self.app.callback([Output('output-2', 'children'), Output('btn-training', 'disabled')], Input('input-trainings', 'value'))
        def createNewSamples(value):
            if value == None or value <= 0:
                self.state.update({'trainings': 0})
                return [('No trainings')], True
            else:
                self.state.update({'trainings': value})
                return [('{} times').format(value)], False

        @self.app.callback(Output('contour-plot', 'figure'), [Input('btn-training', 'n_clicks'), Input('btn-new-samples', 'n_clicks')])
        def drawPlot(n_clicks_training, n_clicks_new_samples):

            if n_clicks_training == 0 and n_clicks_new_samples != 0:
                print('display scatter')
            elif n_clicks_training != 0:
                print('display all')
            else:
                print('display nothing')

            neu = neuron.Neuron([1, 0.5], aft.HeaviSideStepFunction)

            li1 = list(zip(self.state.get('samples1')[
                       0], self.state.get('samples1')[1]))

            for i in range(self.state.get('trainings')):
                for index, trainingTouple in enumerate(li1):
                    neu.train(trainingTouple, 0)
                    neu.updateWeights()

            li2 = list(zip(self.state.get('samples2')[
                       0], self.state.get('samples2')[1]))

            for i in range(self.state.get('trainings')):
                for index, trainingTouple in enumerate(li2):
                    neu.train(trainingTouple, 1)
                    neu.updateWeights()

            x = np.arange(0, 1.01, .01)
            y = x.copy()

            z = []

            for _y in y:
                _z = []
                for _x in x:
                    _z.append(neu.examine([_x, _y]))
                z.append(_z)

            fig = {}

            contour = None

            if n_clicks_training > 0:
                contour = go.Contour(
                    z=z,
                    x=x,
                    y=y
                )

                scatter_x = np.concatenate(
                    (self.state.get('samples1')[0], self.state.get('samples2')[0]))
                scatter_y = np.concatenate(
                    (self.state.get('samples1')[1], self.state.get('samples2')[1]))
                scatter = go.Scatter(
                    x=scatter_x,
                    y=scatter_y,
                    mode='markers',
                    marker=dict(
                        color=[np.concatenate((np.full(20, 0, dtype=int), np.full(20, 1, dtype=int)))])
                )

                # x = np.add(self.state.get('samples1')[
                #     0], self.state.get('samples2')[0]),
                # y = np.add(self.state.get('samples1')[
                #     1], self.state.get('samples2')[1]),
                # print('x', np.add(self.state.get('samples1')
                #                   [0], self.state.get('samples2')[0]))
                # print('y', y)

                fig = go.Figure(data=[scatter, contour])
                fig.update_xaxes(range=[-.1, 1.1])
                fig.update_yaxes(range=[-.1, 1.1])

            return fig


dashServer()
