# Transformer from Scratch: Decoder-Only Chatbot 🤖

A complete, custom-built **Decoder-only Transformer** model written in PyTorch from scratch—**without** using PyTorch's `nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Linear`, `nn.Embedding`, or PyTorch Autograd for the model layers. 

Every single layer, projection, activation function, normalization step, and gradient backpropagation has been **manually implemented** using raw tensor operations and custom autograd functions.

---

## 🚀 Key Features

* **Manual Autograd & Backpropagation**: Hand-coded backward passes (Jacobians) for Self-Attention, SwiGLU Feed-Forward, LayerNorm, Embeddings, and the Output Projection.
* **SwiGLU Activation**: Implements the state-of-the-art gated SwiGLU activation function (used in Llama and PaLM) inside the Feed-Forward network.
* **Weight Tying**: Shares parameters between the input embedding table and the output projection layer to reduce memory footprint and improve generalization.
* **Custom Optimizer**: Built-in **ManualAdamW** optimizer featuring momentum, adaptive learning rates, and decoupled weight decay.
* **Conversational Interface**: Designed to train and run as a chatbot using structured dialogue formatting: `<bos> <user> [prompt] <assistant> [response] <eos>`.

---

## 📊 Model Architecture

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **~8.3 Million** | Capacity adjusted for basic conversational coherence |
| **Model Dimension ($d$)** | **256** | Size of token representation vectors |
| **Layers** | **4** | Stacked custom Decoder blocks |
| **Attention Heads ($h$)** | **8** | Parallel attention subspaces |
| **Head Dimension ($d_k$)** | **32** | Query/Key/Value dimension ($d/h$) |
| **FFN Hidden Dimension** | **1024** | Intermediate feed-forward expansion size |
| **Max Sequence Length** | **256** | Context window size |
| **Vocabulary Size** | **16,160** | Vocabulary range (includes special tokens) |

---

## 📁 Repository Structure

```bash
├── apps/
│   ├── train_decoder.py        # Core training loop & dataset loader
│   ├── server.py               # Fast API inference server for chatbot interactions
│   └── diagnose_model.py       # Helper script to inspect tokenizations & model outputs
├── models/
│   ├── decoder_only_transformer.py  # Model compilation & output projection
│   ├── test_decoder_evaluation.py   # Validation and perplexity evaluation scripts
│   └── test_decoder_only_transformer.py # Architecture tests
├── utils/
│   ├── attention.py            # Causal Multi-Head Self-Attention (Manual Autograd)
│   ├── feed_forward.py         # SwiGLU Feed-Forward network (Manual Autograd)
│   ├── add_and_normalization.py# LayerNorm and residual connection (Manual Autograd)
│   ├── shifted_output_embedder.py # Shift-right, Embeddings, & Positional Encoding (Manual Autograd)
│   └── optimizer.py            # Custom ManualAdamW optimizer implementation
└── documentation/
    └── decoder_training_details.html # High-fidelity math, diagrams, & training walkthrough
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DxZeroHP/Transformer-from-scratch.git
   cd Transformer-from-scratch
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch pandas fastapi uvicorn
   ```

---

## 📈 Training the Model

To run the training process, place your formatted training and validation CSV files under `data/`, and run:

```bash
python apps/train_decoder.py
```

### Prompt-Response Formatting
The training dataset parses sequences structured under the following standard chat template:
```text
<bos> <user> {user_message} <assistant> {assistant_response} <eos>
```

---

## 🖥️ Running the Chatbot Server

Once a model checkpoint is trained and saved as `models/decoder_chatbot.pth`, you can spin up the local inference server:

```bash
python apps/server.py
```

Use the corresponding `apps/index.html` frontend or send POST requests to the API server to chat with the model!

---

## 📖 Under the Hood: Detailed Math & Diagrams

For an in-depth mathematical walkthrough of every layer (including the partial derivatives of the backward passes, positional encodings, and AdamW updates), check out the interactive documentation:

👉 Open [`documentation/decoder_training_details.html`](documentation/decoder_training_details.html) in your web browser.
