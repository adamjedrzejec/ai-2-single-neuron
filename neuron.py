from enum import Enum
import numpy as np
import functionsDerivatives as fd


class ActivationFunctionTypes(Enum):
    HeaviSideStepFunction = 'Heavi Side Function'
    LogisticFunction = 2
    Sin = 3
    Tanh = 4
    Sign = 5
    ReLu = 6
    LeakyReLu = 7


class Neuron():
    def __init__(self, weights, activationFunction):
        self.weights = weights
        self.nextWeights = 'undefined'
        self.theta = np.random.uniform(0.5, 1)
        self.setFunction(activationFunction)

    def setFunction(self, activationFunction):
        if activationFunction == ActivationFunctionTypes.HeaviSideStepFunction:
            self.activationFunction = fd.heaviSideStepFunction
            self.activationDerivative = fd.heaviSideStepFunctionDerivative
        elif activationFunction == ActivationFunctionTypes.LogisticFunction:
            self.activationFunction = fd.logisticFunction
            self.activationDerivative = fd.logisticFunctionDerivative
        else:
            print('Function not yet supported')

    def train(self, X, expected):
        state = np.dot(np.transpose(self.weights), X)
        print('state', state)

        print('output', self.activationFunction(state))

        self.nextWeights = []

        for x in X:
            errorDiff = expected - self.activationFunction(state)
            self.nextWeights.append(self.theta *
                                    (errorDiff) * self.activationDerivative(state) * x)

        print('    weights', self.weights)
        print('nextweights', self.nextWeights)

    def updateWeights(self):
        # pylint: disable=access-member-before-definition
        if self.nextWeights != 'undefined':
            # pylint: disable=access-member-before-definition
            self.weights = np.add(self.weights, self.nextWeights)
            self.nextWeights = 'undefined'
