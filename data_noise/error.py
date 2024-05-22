import numpy as np

def classical_error(p, n):
    return np.random.choice([0,1], size = n, p = [1-p,p])