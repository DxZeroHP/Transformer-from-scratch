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
            "PyTorch build before using MultiHeadCrossAttention."
        )


class MultiHeadCrossAttention:
    def __init__(
        self,
        model_dim,
        num_heads,
        max_query_length=2048,
        max_context_length=2048,
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
        if max_query_length <= 0 or max_context_length <= 0:
            raise ValueError("maximum sequence lengths must be positive")

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.max_query_length = int(max_query_length)
        self.max_context_length = int(max_context_length)
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
        kwargs = {
            "device": self.device,
            "dtype": self.dtype,
            "requires_grad": False,
        }
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

    def _check_input(self, tensor, name, max_length):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(
                f"{name} must have shape (batch, sequence_length, model_dim)"
            )
        if tensor.shape[-1] != self.model_dim:
            raise ValueError(
                f"{name} must have model_dim={self.model_dim}, "
                f"got {tensor.shape[-1]}"
            )
        if tensor.shape[1] > max_length:
            raise ValueError(
                f"{name} sequence length {tensor.shape[1]} exceeds {max_length}"
            )
        return tensor.to(device=self.device, dtype=self.dtype).detach()

    def _split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape
        return (
            tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, tensor):
        batch_size, _, sequence_length, _ = tensor.shape
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.model_dim)
        )

    def _prepare_attention_mask(
        self,
        attention_mask,
        batch_size,
        query_length,
        context_length,
    ):
        if attention_mask is None:
            return None

        mask = torch.as_tensor(
            attention_mask,
            device=self.device,
            dtype=torch.bool,
        )
        try:
            mask = torch.broadcast_to(
                mask,
                (
                    batch_size,
                    self.num_heads,
                    query_length,
                    context_length,
                ),
            )
        except RuntimeError as error:
            raise ValueError(
                "attention_mask must be broadcastable to "
                f"({batch_size}, {self.num_heads}, "
                f"{query_length}, {context_length})"
            ) from error

        if not bool(mask.any(dim=-1).all()):
            raise ValueError(
                "attention_mask must leave at least one context token visible "
                "for every query and head"
            )
        return mask

    def forward(self, query, context, attention_mask=None):
        query = self._check_input(
            query,
            "query",
            self.max_query_length,
        )
        context = self._check_input(
            context,
            "context",
            self.max_context_length,
        )
        if query.shape[0] != context.shape[0]:
            raise ValueError("query and context batch sizes must match")

        batch_size, query_length, _ = query.shape
        context_length = context.shape[1]

        q_linear = query @ self.Wq + self.bq
        k_linear = context @ self.Wk + self.bk
        v_linear = context @ self.Wv + self.bv

        q = self._split_heads(q_linear)
        k = self._split_heads(k_linear)
        v = self._split_heads(v_linear)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = self._prepare_attention_mask(
            attention_mask,
            batch_size,
            query_length,
            context_length,
        )
        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )

        probs = torch.softmax(scores, dim=-1)
        attended = probs @ v
        merged_attended = self._merge_heads(attended)
        output = merged_attended @ self.Wo + self.bo

        self.cache = {
            "query": query,
            "context": context,
            "q": q,
            "k": k,
            "v": v,
            "probs": probs,
            "merged_attended": merged_attended,
            "attention_mask": mask,
            "scale": scale,
        }
        return output

    def __call__(self, query, context, attention_mask=None):
        return self.forward(
            query,
            context,
            attention_mask=attention_mask,
        )

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        dout = self._check_input(
            dout,
            "dout",
            self.max_query_length,
        )
        query = self.cache["query"]
        context = self.cache["context"]
        if dout.shape != query.shape:
            raise ValueError(
                f"dout shape must be {tuple(query.shape)}, "
                f"got {tuple(dout.shape)}"
            )

        q = self.cache["q"]
        k = self.cache["k"]
        v = self.cache["v"]
        probs = self.cache["probs"]
        merged_attended = self.cache["merged_attended"]
        mask = self.cache["attention_mask"]
        scale = self.cache["scale"]

        query_flat = query.reshape(-1, self.model_dim)
        context_flat = context.reshape(-1, self.model_dim)
        dout_flat = dout.reshape(-1, self.model_dim)
        merged_flat = merged_attended.reshape(-1, self.model_dim)

        self.grads["Wo"] += merged_flat.transpose(0, 1) @ dout_flat
        self.grads["bo"] += dout_flat.sum(dim=0)

        dmerged = dout @ self.Wo.transpose(0, 1)
        dattended = self._split_heads(dmerged)

        dprobs = dattended @ v.transpose(-2, -1)
        dv = probs.transpose(-2, -1) @ dattended

        softmax_projection = (dprobs * probs).sum(
            dim=-1,
            keepdim=True,
        )
        dscores = probs * (dprobs - softmax_projection)
        if mask is not None:
            dscores = dscores.masked_fill(~mask, 0)

        dq = (dscores @ k) * scale
        dk = (dscores.transpose(-2, -1) @ q) * scale

        dq_linear = self._merge_heads(dq)
        dk_linear = self._merge_heads(dk)
        dv_linear = self._merge_heads(dv)

        dq_flat = dq_linear.reshape(-1, self.model_dim)
        dk_flat = dk_linear.reshape(-1, self.model_dim)
        dv_flat = dv_linear.reshape(-1, self.model_dim)

        self.grads["Wq"] += query_flat.transpose(0, 1) @ dq_flat
        self.grads["Wk"] += context_flat.transpose(0, 1) @ dk_flat
        self.grads["Wv"] += context_flat.transpose(0, 1) @ dv_flat
        self.grads["bq"] += dq_flat.sum(dim=0)
        self.grads["bk"] += dk_flat.sum(dim=0)
        self.grads["bv"] += dv_flat.sum(dim=0)

        dquery = dq_linear @ self.Wq.transpose(0, 1)
        dcontext = (
            dk_linear @ self.Wk.transpose(0, 1)
            + dv_linear @ self.Wv.transpose(0, 1)
        )
        return dquery, dcontext

    def step(self, learning_rate, weight_decay=0.0):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        for name, parameter in self.parameters().items():
            grad = self.grads[name]
            if weight_decay:
                grad = grad + weight_decay * parameter
            parameter -= learning_rate * grad

    def to(self, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        self.device = device
        self.dtype = dtype

        for name in list(self.parameters().keys()):
            setattr(
                self,
                name,
                getattr(self, name).to(device=device, dtype=dtype),
            )
        for name in list(self.grads.keys()):
            self.grads[name] = self.grads[name].to(
                device=device,
                dtype=dtype,
            )

        self.cache = None
        return self

    def state_dict(self):
        return {
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "max_query_length": self.max_query_length,
            "max_context_length": self.max_context_length,
            "weights": {
                name: tensor.detach().cpu()
                for name, tensor in self.parameters().items()
            },
        }

    def load_state_dict(self, state):
        if int(state["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this module")
        if int(state["num_heads"]) != self.num_heads:
            raise ValueError("state num_heads does not match this module")

        for name, tensor in state["weights"].items():
            setattr(
                self,
                name,
                tensor.to(device=self.device, dtype=self.dtype),
            )

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
            max_query_length=int(state["max_query_length"]),
            max_context_length=int(state["max_context_length"]),
            device=device,
            dtype=dtype,
        )
        obj.load_state_dict(state)
        return obj


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = MultiHeadCrossAttention(
        model_dim=256,
        num_heads=4,
        max_query_length=128,
        max_context_length=256,
        device=device,
        dtype=torch.float32,
        seed=42,
    )

    query = torch.randn(2, 16, 256, device=device)
    context = torch.randn(2, 32, 256, device=device)
    output = attention(query, context)
    dquery, dcontext = attention.backward(torch.ones_like(output))

    print("Device:", output.device)
    print("Output shape:", tuple(output.shape))
    print("Query gradient shape:", tuple(dquery.shape))
    print("Context gradient shape:", tuple(dcontext.shape))
