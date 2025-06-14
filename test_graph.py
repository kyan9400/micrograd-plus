from core.engine import Value
from optim.optimizers import Adam
from utils.visualizer import draw_dot

# Simple linear model: y = w * x + b
w = Value(2.0, label='w')
b = Value(-1.0, label='b')

x = 3.0          # input value
target = 7.0     # desired output

# Setup parameters and optimizer
params = [w, b]
optimizer = Adam(params, lr=0.01)

# Training loop
for step in range(20):
    # Forward pass
    y = w * x + b
    y.label = 'y'

    loss = (y - target) * (y - target)
    loss.label = 'loss'

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Reset gradients for next iteration
    optimizer.zero_grad()

    print(f"Step {step}: y = {y.data:.4f}, loss = {loss.data:.4f}")

# ✅ Visualize the final computation graph
dot = draw_dot(loss)
dot.render('computation_graph', view=True)
