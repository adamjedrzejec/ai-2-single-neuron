import neuron
from neuron import ActivationFunctionTypes as aft
import classifier as c

neu = neuron.Neuron([1, 0.5], aft.LogisticFunction)

for i in range(10):
    neu.train([1, -1], 0.25)
    neu.updateWeights()
    print()

c1 = c.Classifier(2, 5)
x, y = c1.getAllSamples()

print(x, y)
