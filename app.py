import gradio as gr
from core.engine import Value
from core.mlp import MLP
import math

# Load pre-trained sine model
sine_model = MLP(1, [16, 16, 1])
# Optionally: load trained weights from file in future

# Train XOR model quickly (can replace with saved model later)
xor_model = MLP(2, [4, 1])
xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]
from optim.optimizers import Adam
optimizer = Adam(xor_model.parameters(), lr=0.1)
for _ in range(100):
    loss = Value(0.0)
    for x, y in xor_data:
        pred = xor_model([Value(xi) for xi in x])
        loss += (pred - y) * (pred - y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === Inference functions ===
def predict_sine(x: float):
    v = Value(x)
    out = sine_model([v])
    return out.data

def predict_xor(x1: float, x2: float):
    out = xor_model([Value(x1), Value(x2)])
    return out.data

# === Gradio Interface ===

with gr.Blocks() as demo:
    gr.Markdown("## 🔢 Micrograd+ Demo: Sine & XOR Inference")

    with gr.Tab("🔁 XOR"):
        x1 = gr.Slider(0, 1, step=1, label="x1")
        x2 = gr.Slider(0, 1, step=1, label="x2")
        xor_out = gr.Number(label="Prediction")
        xor_btn = gr.Button("Predict XOR")
        xor_btn.click(predict_xor, inputs=[x1, x2], outputs=xor_out)

    with gr.Tab("🌊 Sine"):
        x_sin = gr.Slider(-math.pi, math.pi, step=0.01, label="x")
        sin_out = gr.Number(label="sin(x) prediction")
        sin_btn = gr.Button("Predict Sine")
        sin_btn.click(predict_sine, inputs=[x_sin], outputs=sin_out)

# === Launch the app ===
demo.launch()
