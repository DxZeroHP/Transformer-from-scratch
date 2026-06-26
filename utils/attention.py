import math
import os

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for GPU tensor math. Install a CUDA-enabled "
            "PyTorch build before using MultiHeadSelfAttention."
        )


class MultiHeadSelfAttention:
    def __init__(
        self,
        model_dim,
        num_heads,
        max_sequence_length=2048,
        device=None,
        dtype=None,
        seed=None,
    ):
        _require_torch()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.max_sequence_length = int(max_sequence_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None

        self.Wq = self._init_weight(generator)
        self.Wk = self._init_weight(generator)
        self.Wv = self._init_weight(generator)
        self.Wo = self._init_weight(generator)

        self.bq = self._zeros((self.model_dim,))
        self.bk = self._zeros((self.model_dim,))
        self.bv = self._zeros((self.model_dim,))
        self.bo = self._zeros((self.model_dim,))

        self.cache = None
        self.zero_grad()

    def _randn(self, shape, generator):
        kwargs = {"device": self.device, "dtype": self.dtype, "requires_grad": False}
        if generator is not None:
            kwargs["generator"] = generator
        return torch.randn(shape, **kwargs)

    def _zeros(self, shape):
        return torch.zeros(
            shape,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )

    def _init_weight(self, generator):
        scale = 1.0 / math.sqrt(self.model_dim)
        return self._randn((self.model_dim, self.model_dim), generator) * scale

    def parameters(self):
        return {
            "Wq": self.Wq,
            "Wk": self.Wk,
            "Wv": self.Wv,
            "Wo": self.Wo,
            "bq": self.bq,
            "bk": self.bk,
            "bv": self.bv,
            "bo": self.bo,
        }

    def zero_grad(self):
        self.grads = {
            "Wq": self._zeros((self.model_dim, self.model_dim)),
            "Wk": self._zeros((self.model_dim, self.model_dim)),
            "Wv": self._zeros((self.model_dim, self.model_dim)),
            "Wo": self._zeros((self.model_dim, self.model_dim)),
            "bq": self._zeros((self.model_dim,)),
            "bk": self._zeros((self.model_dim,)),
            "bv": self._zeros((self.model_dim,)),
            "bo": self._zeros((self.model_dim,)),
        }

    def _check_input(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence_length, model_dim)")
        if x.shape[-1] != self.model_dim:
            raise ValueError(f"expected model_dim={self.model_dim}, got {x.shape[-1]}")
        if x.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )
        if x.device.type != torch.device(self.device).type:
            x = x.to(self.device)
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x.detach()

    def _split_heads(self, x):
        batch_size, sequence_length, _ = x.shape
        return (
            x.view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, x):
        batch_size, _, sequence_length, _ = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.model_dim)
        )

    def _causal_mask(self, sequence_length):
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def forward(self, x, causal=True, attention_mask=None):
        x = self._check_input(x)
        batch_size, sequence_length, _ = x.shape

        q_linear = x @ self.Wq + self.bq
        k_linear = x @ self.Wk + self.bk
        v_linear = x @ self.Wv + self.bv

        q = self._split_heads(q_linear)
        k = self._split_heads(k_linear)
        v = self._split_heads(v_linear)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = None
        if causal:
            mask = self._causal_mask(sequence_length)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        context = probs @ v
        merged_context = self._merge_heads(context)
        out = merged_context @ self.Wo + self.bo

        self.cache = {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "probs": probs,
            "context": context,
            "merged_context": merged_context,
            "causal_mask": mask,
            "attention_mask": attention_mask,
            "scale": scale,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        }
        return out

    def __call__(self, x, causal=True, attention_mask=None):
        return self.forward(x, causal=causal, attention_mask=attention_mask)

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        dout = dout.to(device=self.device, dtype=self.dtype).detach()
        x = self.cache["x"]
        q = self.cache["q"]
        k = self.cache["k"]
        v = self.cache["v"]
        probs = self.cache["probs"]
        merged_context = self.cache["merged_context"]
        scale = self.cache["scale"]
        batch_size = self.cache["batch_size"]
        sequence_length = self.cache["sequence_length"]

        x_flat = x.reshape(-1, self.model_dim)
        dout_flat = dout.reshape(-1, self.model_dim)
        merged_flat = merged_context.reshape(-1, self.model_dim)

        self.grads["Wo"] += merged_flat.transpose(0, 1) @ dout_flat
        self.grads["bo"] += dout_flat.sum(dim=0)

        dmerged = dout @ self.Wo.transpose(0, 1)
        dcontext = self._split_heads(dmerged)

        dprobs = dcontext @ v.transpose(-2, -1)
        dv = probs.transpose(-2, -1) @ dcontext

        dsoftmax_sum = (dprobs * probs).sum(dim=-1, keepdim=True)
        dscores = probs * (dprobs - dsoftmax_sum)

        causal_mask = self.cache["causal_mask"]
        if causal_mask is not None:
            dscores = dscores.masked_fill(causal_mask, 0)

        attention_mask = self.cache["attention_mask"]
        if attention_mask is not None:
            dscores = dscores.masked_fill(~attention_mask, 0)

        dq = (dscores @ k) * scale
        dk = (dscores.transpose(-2, -1) @ q) * scale

        dq_linear = self._merge_heads(dq)
        dk_linear = self._merge_heads(dk)
        dv_linear = self._merge_heads(dv)

        dq_flat = dq_linear.reshape(batch_size * sequence_length, self.model_dim)
        dk_flat = dk_linear.reshape(batch_size * sequence_length, self.model_dim)
        dv_flat = dv_linear.reshape(batch_size * sequence_length, self.model_dim)

        self.grads["Wq"] += x_flat.transpose(0, 1) @ dq_flat
        self.grads["Wk"] += x_flat.transpose(0, 1) @ dk_flat
        self.grads["Wv"] += x_flat.transpose(0, 1) @ dv_flat
        self.grads["bq"] += dq_flat.sum(dim=0)
        self.grads["bk"] += dk_flat.sum(dim=0)
        self.grads["bv"] += dv_flat.sum(dim=0)

        dx = (
            dq_linear @ self.Wq.transpose(0, 1)
            + dk_linear @ self.Wk.transpose(0, 1)
            + dv_linear @ self.Wv.transpose(0, 1)
        )
        return dx

    def step(self, learning_rate, weight_decay=0.0, grad_clip=None):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        for name, parameter in self.parameters().items():
            grad = self.grads[name]
            if grad_clip is not None:
                grad = grad.clamp(min=-grad_clip, max=grad_clip)
            if weight_decay:
                grad = grad + weight_decay * parameter
            parameter -= learning_rate * grad

    def to(self, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        self.device = device
        self.dtype = dtype

        for name in list(self.parameters().keys()):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        for name in list(self.grads.keys()):
            self.grads[name] = self.grads[name].to(device=device, dtype=dtype)
        if self.cache is not None:
            self.cache = None
        return self

    def state_dict(self):
        state = {
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "max_sequence_length": self.max_sequence_length,
            "weights": {name: tensor.detach().cpu() for name, tensor in self.parameters().items()},
        }
        return state

    def load_state_dict(self, state):
        if int(state["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this module")
        if int(state["num_heads"]) != self.num_heads:
            raise ValueError("state num_heads does not match this module")

        for name, tensor in state["weights"].items():
            setattr(self, name, tensor.to(device=self.device, dtype=self.dtype))
        self.zero_grad()
        self.cache = None

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device=None, dtype=None):
        _require_torch()
        state = torch.load(path, map_location="cpu")
        obj = cls(
            model_dim=int(state["model_dim"]),
            num_heads=int(state["num_heads"]),
            max_sequence_length=int(state["max_sequence_length"]),
            device=device,
            dtype=dtype,
        )
        obj.load_state_dict(state)
        return obj


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = MultiHeadSelfAttention(
        model_dim=256,
        num_heads=4,
        max_sequence_length=128,
        device=device,
        dtype=torch.float32,
        seed=42,
    )
    x = torch.randn(2, 16, 256, device=device)
    out = attention(x)
    dx = attention.backward(torch.ones_like(out))
    print("Device:", device)
    print("Output shape:", tuple(out.shape))
    print("Input gradient shape:", tuple(dx.shape))
