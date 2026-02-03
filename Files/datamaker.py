import random
import math
import numpy as np

def generate(samples):
    random.seed(42)
    inputs = []
    outputs = []
    for s in range(samples):
        x1 = random.uniform(0,5)
        x2 = random.uniform(0,5)

        x1_normalized = x1 / 5

        if x1_normalized < 0.4:
            y = math.sin(x1) + math.cos(x2)
        elif x1_normalized > 0.4 and x1_normalized < 0.6:
            y = (1 - x1_normalized) * math.sin(x1) + math.cos(x2) + (x1_normalized) * math.sin(x1 * x2) + math.cos(x1 - x2)
        else:
            y = math.sin(x1 * x2) + math.cos(x1 - x2)

        inputs.append([x1, x2])
        outputs.append([y])
    np.save("Inputs.npy", inputs)
    np.save("Outputs.npy", outputs)

generate(10000)
