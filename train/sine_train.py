import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.mlp import MLP
from optim.optimizers import Adam
from data.sine import load_sine
from core.engine import Value
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange  # progress bar

# Load training data
data = load_sine(100)  # use fewer samples for speed

# Create model: 1 input → hidden layers → 1 output
model = MLP(1, [16, 16, 1])
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in trange(100):  # fewer epochs = faster
    total_loss = Value(0.0)
    for x_vals, y_target in data:
        x = [Value(v) for v in x_vals]
        y_pred = model(x)
        loss = (y_pred - y_target) * (y_pred - y_target)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 99:
        print(f"\nEpoch {epoch}: loss = {total_loss.data:.4f}")

# ✅ After training: plot predictions
x_test = np.linspace(-np.pi, np.pi, 100)
y_true = np.sin(x_test)
y_pred = [model([Value(x)]).data for x in x_test]

plt.figure(figsize=(8, 4))
plt.plot(x_test, y_true, label="True sin(x)", linewidth=2)
plt.plot(x_test, y_pred, label="MLP prediction", linestyle="--")
plt.title("Sine Wave Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
