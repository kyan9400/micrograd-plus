import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import Value
from optim.optimizers import Adam
from data.xor import load_xor
import random

# Activation function
def tanh(x):
    return x.tanh()

# Single neuron layer
def neuron(x, w, b):
    act = sum([xi * wi for xi, wi in zip(x, w)]) + b
    return tanh(act)

# XOR MLP: 2-2-1
# Layer 1 weights and biases (2 hidden neurons)
w1 = [Value(random.uniform(-1, 1)) for _ in range(2)]
w2 = [Value(random.uniform(-1, 1)) for _ in range(2)]
b1 = Value(0.0)
b2 = Value(0.0)

# Output layer weights (1 neuron)
w_out = [Value(random.uniform(-1, 1)) for _ in range(2)]
b_out = Value(0.0)

# All parameters
parameters = w1 + w2 + [b1, b2] + w_out + [b_out]
optimizer = Adam(parameters, lr=0.05)

# Load XOR data
data = load_xor()

# Training loop
for epoch in range(200):
    total_loss = Value(0.0)

    for x_vals, target in data:
        x = [Value(v) for v in x_vals]

        h1 = neuron(x, w1, b1)
        h2 = neuron(x, w2, b2)

        out = neuron([h1, h2], w_out, b_out)
        loss = (out - target) * (out - target)
        total_loss = total_loss + loss

    # Backward + Update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 199:
        print(f"Epoch {epoch}: loss = {total_loss.data:.4f}")

# Final evaluation
print("\nFinal predictions:")
for x_vals, _ in data:
    x = [Value(v) for v in x_vals]
    h1 = neuron(x, w1, b1)
    h2 = neuron(x, w2, b2)
    out = neuron([h1, h2], w_out, b_out)
    print(f"Input: {x_vals} → Pred: {out.data:.4f}")
