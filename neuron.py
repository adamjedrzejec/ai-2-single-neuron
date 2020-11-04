from enum import Enum


class ActivationFunctionTypes(Enum):
    HeaviSideStepFunction = 1
    LogisticFunction = 2
    Sin = 3
    Tanh = 4
    Sign = 5
    ReLu = 6
    LeakyReLu = 7


class Neuron():
    def __init__(self, weights, activationFunction):
        self.weights = weights
        self.activationFunction = activationFunction
