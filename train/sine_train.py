import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.mlp import MLP
from optim.optimizers import Adam
from data.sine import load_sine
from core.engine import Value

# Load data
data = load_sine(200)

# Create model: 1 input → [16, 16] hidden → 1 output
model = MLP(1, [16, 16, 1])
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    total_loss = Value(0.0)
    for x_vals, y_target in data:
        x = [Value(v) for v in x_vals]
        y_pred = model(x)
        loss = (y_pred - y_target) * (y_pred - y_target)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 199:
        print(f"Epoch {epoch}: loss = {total_loss.data:.4f}")
