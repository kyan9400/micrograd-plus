import math
import random

def load_sine(n=100):
    data = []
    for _ in range(n):
        x = random.uniform(-math.pi, math.pi)
        y = math.sin(x)
        data.append(([x], y))  # input is 1D
    return data
