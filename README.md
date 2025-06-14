# 🧠 micrograd-plus

**An extended educational project based on Karpathy’s [micrograd](https://github.com/karpathy/micrograd)** — now with a visual computation graph, optimizers like SGD/Adam, and support for toy datasets like XOR and sine wave fitting.

---

## 🚀 Features

- ✅ Tiny autodiff engine with forward/backward pass
- ✅ Computation graph visualizer (Graphviz)
- ✅ Optimizers: SGD (Adam coming soon)
- ✅ Toy datasets (XOR, sine, classification blobs - WIP)
- ✅ Modular structure for extending neurons, MLPs, training demos

---

## 📁 Folder Structure

```
micrograd-plus/
├── core/           # Autograd engine
│   └── engine.py
├── optim/          # Optimizers like SGD
│   └── optimizers.py
├── utils/          # Visualizer
│   └── visualizer.py
├── data/           # Toy dataset generators (planned)
├── test_graph.py   # Demo & training script
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/kyan9400/micrograd-plus.git
cd micrograd-plus

python -m venv venv
.env\Scriptsctivate       # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

If you don't have `Graphviz` installed on your system, download it from [https://graphviz.org/download](https://graphviz.org/download) and add it to your `PATH`.

---

## 🧪 Run Demo

```bash
python test_graph.py
```

This trains a simple 1D model: `y = wx + b`, and visualizes the graph if enabled.

---

## 📊 Visualize the Computation Graph

To visualize the graph with values and gradients:

```python
from utils.visualizer import draw_dot

dot = draw_dot(loss)
dot.render('computation_graph', view=True)
```

---

## 🔄 Optimizers

Currently supported:
- ✅ SGD (stochastic gradient descent)

Coming soon:
- [ ] Adam
- [ ] Momentum
- [ ] RMSProp

---

## 📚 Credits

Inspired by [@karpathy/micrograd](https://github.com/karpathy/micrograd)  
Extended and customized by [@kyan9400](https://github.com/kyan9400)

---

## 📌 License

MIT License
