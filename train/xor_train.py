import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.mlp import MLP
from optim.optimizers import Adam
from data.xor import load_xor
from core.engine import Value

# Load XOR data
data = load_xor()

# Create MLP: 2 inputs → 4 hidden → 1 output
model = MLP(2, [4, 1])
optimizer = Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    total_loss = Value(0.0)

    for x_vals, y_target in data:
        x = [Value(v) for v in x_vals]
        y_pred = model(x)
        loss = (y_pred - y_target) * (y_pred - y_target)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}: loss = {total_loss.data:.4f}")

# Final evaluation
print("\nFinal predictions:")
for x_vals, _ in data:
    x = [Value(v) for v in x_vals]
    out = model(x)
    print(f"Input: {x_vals} → Pred: {out.data:.4f}")
