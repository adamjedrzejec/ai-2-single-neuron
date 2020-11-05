import neuron
from neuron import ActivationFunctionTypes as aft

neu = neuron.Neuron([1, 0.5], aft.LogisticFunction)

for i in range(10):
    neu.train([1, -1], 0.25)
    neu.updateWeights()
    print()
