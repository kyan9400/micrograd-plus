import gradio as gr
from core.engine import Value
from core.mlp import MLP
from optim.optimizers import Adam
import math
import numpy as np
import matplotlib.pyplot as plt

# === Train quick XOR model ===
xor_model = MLP(2, [4, 1])
xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]
optimizer = Adam(xor_model.parameters(), lr=0.1)
for _ in range(100):
    loss = Value(0.0)
    for x, y in xor_data:
        pred = xor_model([Value(xi) for xi in x])
        loss += (pred - y) * (pred - y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === Train quick sine model ===
sine_model = MLP(1, [16, 16, 1])
sine_data = [([x], math.sin(x)) for x in np.linspace(-math.pi, math.pi, 100)]
optimizer = Adam(sine_model.parameters(), lr=0.01)
for _ in range(100):
    loss = Value(0.0)
    for x, y in sine_data:
        pred = sine_model([Value(xi) for xi in x])
        loss += (pred - y) * (pred - y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === Inference functions ===
def predict_xor(x1: float, x2: float):
    out = xor_model([Value(x1), Value(x2)])
    return out.data

def predict_sine_with_plot(x_val: float):
    x_range = np.linspace(-math.pi, math.pi, 100)
    y_true = np.sin(x_range)
    y_pred = [sine_model([Value(x)]).data for x in x_range]
    y_point = sine_model([Value(x_val)]).data

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_range, y_true, label="True sin(x)", linewidth=2)
    ax.plot(x_range, y_pred, label="MLP prediction", linestyle="--")
    ax.scatter([x_val], [y_point], color="red", label="Input x")
    ax.set_title("Sine Curve Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    return fig, y_point

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Micrograd+ Demo: XOR + Sine Plot")

    with gr.Tab("🔁 XOR"):
        x1 = gr.Slider(0, 1, step=1, label="x1")
        x2 = gr.Slider(0, 1, step=1, label="x2")
        xor_out = gr.Number(label="XOR Prediction")
        xor_btn = gr.Button("Predict")
        xor_btn.click(predict_xor, inputs=[x1, x2], outputs=xor_out)

    with gr.Tab("🌊 Sine"):
        x_slider = gr.Slider(-math.pi, math.pi, step=0.05, label="x input")
        y_output = gr.Number(label="MLP sin(x) prediction")
        sine_plot = gr.Plot(label="Sine vs MLP Prediction")
        x_slider.change(predict_sine_with_plot, inputs=[x_slider], outputs=[sine_plot, y_output])

demo.launch()
