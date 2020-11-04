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
        self.theta = np.random.uniform()
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

    def train(self):
        self.nextWeights = self.theta * ()

    def updateWeights(self):
        # pylint: disable=access-member-before-definition
        if self.nextWeights != 'undefined':
            # pylint: disable=access-member-before-definition
            self.weights = self.nextWeights
            self.nextWeights = 'undefined'
