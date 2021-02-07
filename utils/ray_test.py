import numpy as np
from execution_time import ExecutionTime
e = ExecutionTime()


@e.timeit
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
