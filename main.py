import neuron
from neuron import ActivationFunctionTypes as aft

ner = neuron.Neuron(1, aft.HeaviSideStepFunction)
ner.updateWeights()
