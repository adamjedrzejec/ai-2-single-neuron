import numpy as np

import neuron
from neuron import ActivationFunctionTypes as aft
import classifier as c

neu = neuron.Neuron([1, 0.5], aft.HeaviSideStepFunction)

# for i in range(10):
#     neu.train([1, -1], 0.25)
#     neu.updateWeights()
#     print()

c1 = c.Classifier(2, 400)
x1, y1 = c1.getAllSamples()

li1 = list(zip(x1, y1))

for i in range(200):
    for index, trainingTouple in enumerate(li1):
        # print(index, trainingTouple)
        neu.train(trainingTouple, 0)
        neu.updateWeights()


c2 = c.Classifier(2, 400)
x2, y2 = c2.getAllSamples()

li2 = list(zip(x2, y2))
for i in range(200):
    for index, trainingTouple in enumerate(li2):
        # print(index, trainingTouple)
        neu.train(trainingTouple, 1)
        neu.updateWeights()


print('shouldBeTrue', neu.examine(li1[2], 0))
print('shouldBeTrue', neu.examine(li2[2], 1))
