
import gradio as gr
from core.engine import Value
from core.mlp import MLP
from optim.optimizers import Adam
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json

# === Utility: Save and load model weights ===
def save_model(model, path):
    os.makedirs("checkpoints", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([p.data for p in model.parameters()], f)

def save_model_as_json(model, path):
    os.makedirs("checkpoints", exist_ok=True)
    data = [p.data for p in model.parameters()]
    with open(path, "w") as f:
        json.dump(data, f)

def load_model(model, path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            weights = pickle.load(f)
        for p, w in zip(model.parameters(), weights):
            p.data = w
        return True
    return False

# === Training logic ===
def train_xor_model(epochs=100, lr=0.1, hidden=4, save=False):
    model = MLP(2, [hidden, 1])
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = Value(0.0)
        for x, y in data:
            pred = model([Value(xi) for xi in x])
            loss += (pred - y) * (pred - y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if save:
        save_model(model, "checkpoints/xor_model.pkl")
        save_model_as_json(model, "checkpoints/xor_model.json")
    return model

def train_sine_model(epochs=100, lr=0.01, hidden1=16, hidden2=16, save=False):
    model = MLP(1, [hidden1, hidden2, 1])
    data = [([x], math.sin(x)) for x in np.linspace(-math.pi, math.pi, 100)]
    optimizer = Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = Value(0.0)
        for x, y in data:
            pred = model([Value(xi) for xi in x])
            loss += (pred - y) * (pred - y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if save:
        save_model(model, "checkpoints/sine_model.pkl")
        save_model_as_json(model, "checkpoints/sine_model.json")
    return model

# === Init models ===
xor_model = train_xor_model(0)
if not load_model(xor_model, "checkpoints/xor_model.pkl"):
    xor_model = train_xor_model(100, save=True)

sine_model = train_sine_model(0)
if not load_model(sine_model, "checkpoints/sine_model.pkl"):
    sine_model = train_sine_model(100, save=True)

# === Inference ===
def predict_xor(x1, x2):
    return xor_model([Value(x1), Value(x2)]).data

def predict_sine(x_val):
    return sine_model([Value(x_val)]).data

def plot_sine(x_val):
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
    return fig

# === Retraining + Load callbacks ===
def retrain_xor(epochs, lr, hidden):
    global xor_model
    xor_model = train_xor_model(epochs=epochs, lr=lr, hidden=hidden, save=True)
    return "✅ XOR model retrained and saved."

def retrain_sine(epochs, lr, h1, h2):
    global sine_model
    sine_model = train_sine_model(epochs=epochs, lr=lr, hidden1=h1, hidden2=h2, save=True)
    data = [([x], math.sin(x)) for x in np.linspace(-math.pi, math.pi, 100)]
    losses = []
    optimizer = Adam(sine_model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = Value(0.0)
        for x, y in data:
            pred = sine_model([Value(xi) for xi in x])
            loss += (pred - y) * (pred - y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            losses.append((epoch, loss.data))

    fig, ax = plt.subplots(figsize=(6, 3))
    epochs_, loss_vals = zip(*losses)
    ax.plot(epochs_, loss_vals, marker='o')
    ax.set_title("Sine Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    return fig, "✅ Sine model retrained and saved."

def load_latest_xor():
    global xor_model
    loaded = load_model(xor_model, "checkpoints/xor_model.pkl")
    return "✅ XOR model loaded." if loaded else "❌ Failed to load XOR model."

def load_latest_sine():
    global sine_model
    loaded = load_model(sine_model, "checkpoints/sine_model.pkl")
    return "✅ Sine model loaded." if loaded else "❌ Failed to load Sine model."

# === UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Micrograd+ Trainer (with JSON Export)")

    with gr.Tab("🔁 XOR Logic"):
        with gr.Row():
            x1 = gr.Slider(0, 1, step=1, label="x1")
            x2 = gr.Slider(0, 1, step=1, label="x2")
            xor_out = gr.Number(label="Prediction")
        xor_btn = gr.Button("Predict XOR")
        xor_btn.click(predict_xor, [x1, x2], xor_out)

        gr.Markdown("### 🔧 Retrain XOR Model")
        xor_epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
        xor_lr = gr.Slider(0.001, 0.5, value=0.1, step=0.01, label="Learning Rate")
        xor_hidden = gr.Slider(2, 16, value=4, step=1, label="Hidden Neurons")
        xor_status = gr.Textbox(label="Status", interactive=False)
        retrain_xor_btn = gr.Button("🔄 Retrain XOR")
        retrain_xor_btn.click(retrain_xor, [xor_epochs, xor_lr, xor_hidden], xor_status)

        load_xor_btn = gr.Button("🔁 Load Saved XOR Model")
        load_xor_btn.click(load_latest_xor, outputs=xor_status)

        xor_download = gr.File(label="Download XOR Model (.json)")
        export_xor_btn = gr.Button("📥 Export XOR as JSON")
        export_xor_btn.click(
            fn=lambda: "checkpoints/xor_model.json",
            outputs=xor_download
        )

    with gr.Tab("🌊 Sine Fitting"):
        with gr.Row():
            sine_x = gr.Slider(-math.pi, math.pi, step=0.05, label="Input x")
            sine_y = gr.Number(label="Predicted sin(x)")
            sine_plot = gr.Plot(label="Prediction Curve")
        sine_x.change(fn=lambda x: (plot_sine(x), predict_sine(x)), inputs=sine_x, outputs=[sine_plot, sine_y])

        gr.Markdown("### 🔧 Retrain Sine Model")
        sine_epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
        sine_lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate")
        sine_h1 = gr.Slider(2, 64, value=16, step=2, label="Hidden Layer 1")
        sine_h2 = gr.Slider(2, 64, value=16, step=2, label="Hidden Layer 2")
        sine_loss_plot = gr.Plot(label="Training Loss")
        sine_status = gr.Textbox(label="Status", interactive=False)
        retrain_sine_btn = gr.Button("🔄 Retrain Sine")
        retrain_sine_btn.click(retrain_sine, [sine_epochs, sine_lr, sine_h1, sine_h2], [sine_loss_plot, sine_status])

        load_sine_btn = gr.Button("🔁 Load Saved Sine Model")
        load_sine_btn.click(load_latest_sine, outputs=sine_status)

        sine_download = gr.File(label="Download Sine Model (.json)")
        export_sine_btn = gr.Button("📥 Export Sine as JSON")
        export_sine_btn.click(
            fn=lambda: "checkpoints/sine_model.json",
            outputs=sine_download
        )

demo.launch()
