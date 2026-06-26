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
            "PyTorch build before using ShiftedOutputEmbedder."
        )


class ShiftedOutputEmbedder:
    def __init__(
        self,
        vocab_size,
        model_dim,
        bos_token_id,
        pad_token_id=0,
        max_sequence_length=2048,
        device=None,
        dtype=None,
        seed=None,
    ):
        _require_torch()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.max_sequence_length = int(max_sequence_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        self._validate_special_id(self.bos_token_id, "bos_token_id")
        if self.pad_token_id is not None:
            self._validate_special_id(self.pad_token_id, "pad_token_id")

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None

        scale = 1.0 / math.sqrt(self.model_dim)
        self.embedding_table = self._randn(
            (self.vocab_size, self.model_dim),
            generator,
        ) * scale

        if self.pad_token_id is not None:
            self.embedding_table[self.pad_token_id] = 0

        self.position_table = self._build_sinusoidal_positions()
        self.cache = None
        self.zero_grad()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        model_dim,
        max_sequence_length=2048,
        device=None,
        dtype=None,
        seed=None,
    ):
        return cls(
            vocab_size=len(tokenizer.vocab),
            model_dim=model_dim,
            bos_token_id=tokenizer.vocab[tokenizer.bos_token],
            pad_token_id=tokenizer.vocab[tokenizer.pad_token],
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def _validate_special_id(self, token_id, name):
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(
                f"{name} must satisfy 0 <= id < {self.vocab_size}, got {token_id}"
            )

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

    def _build_sinusoidal_positions(self):
        positions = torch.arange(
            self.max_sequence_length,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                self.model_dim,
                2,
                device=self.device,
                dtype=self.dtype,
            )
            * (-math.log(10000.0) / self.model_dim)
        )

        table = self._zeros((self.max_sequence_length, self.model_dim))
        table[:, 0::2] = torch.sin(positions * frequencies)
        odd_width = table[:, 1::2].shape[1]
        table[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
        return table

    def parameters(self):
        return {"embedding_table": self.embedding_table}

    def zero_grad(self):
        self.grads = {
            "embedding_table": self._zeros(
                (self.vocab_size, self.model_dim)
            )
        }

    def _check_token_ids(self, token_ids):
        if not torch.is_tensor(token_ids):
            token_ids = torch.as_tensor(token_ids, dtype=torch.long)

        token_ids = token_ids.to(device=self.device, dtype=torch.long).detach()
        squeeze_batch = False
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_batch = True
        elif token_ids.ndim != 2:
            raise ValueError(
                "target_token_ids must have shape (sequence_length,) or "
                "(batch, sequence_length)"
            )

        sequence_length = token_ids.shape[1]
        if sequence_length == 0:
            raise ValueError("target_token_ids cannot have an empty sequence")
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )

        min_id = int(token_ids.min())
        max_id = int(token_ids.max())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"token id out of range: expected 0 <= id < {self.vocab_size}, "
                f"got min={min_id}, max={max_id}"
            )

        return token_ids, squeeze_batch

    def shift_right(self, target_token_ids):
        target_token_ids, squeeze_batch = self._check_token_ids(
            target_token_ids
        )
        shifted_ids = torch.empty_like(target_token_ids)
        shifted_ids[:, 0] = self.bos_token_id
        shifted_ids[:, 1:] = target_token_ids[:, :-1]

        if squeeze_batch:
            return shifted_ids[0]
        return shifted_ids

    def forward(self, target_token_ids):
        target_token_ids, squeeze_batch = self._check_token_ids(
            target_token_ids
        )

        shifted_ids = torch.empty_like(target_token_ids)
        shifted_ids[:, 0] = self.bos_token_id
        shifted_ids[:, 1:] = target_token_ids[:, :-1]

        sequence_length = shifted_ids.shape[1]
        embeddings = self.embedding_table[shifted_ids] * math.sqrt(
            self.model_dim
        )
        embeddings = embeddings + self.position_table[:sequence_length]

        if self.pad_token_id is None:
            active_mask = torch.ones_like(shifted_ids, dtype=torch.bool)
        else:
            active_mask = shifted_ids != self.pad_token_id
            embeddings = embeddings.masked_fill(
                ~active_mask.unsqueeze(-1),
                0,
            )

        self.cache = {
            "shifted_ids": shifted_ids,
            "active_mask": active_mask,
            "squeeze_batch": squeeze_batch,
            "output_shape": embeddings.shape,
        }

        if squeeze_batch:
            return embeddings[0], shifted_ids[0]
        return embeddings, shifted_ids

    def __call__(self, target_token_ids):
        return self.forward(target_token_ids)

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")
        if not torch.is_tensor(dout):
            raise TypeError("dout must be a torch.Tensor")

        dout = dout.to(device=self.device, dtype=self.dtype).detach()
        if self.cache["squeeze_batch"]:
            if dout.ndim != 2:
                raise ValueError(
                    "dout must have shape (sequence_length, model_dim)"
                )
            dout = dout.unsqueeze(0)

        if dout.shape != self.cache["output_shape"]:
            raise ValueError(
                f"dout shape must be {tuple(self.cache['output_shape'])}, "
                f"got {tuple(dout.shape)}"
            )

        shifted_ids = self.cache["shifted_ids"]
        active_mask = self.cache["active_mask"]
        scale = math.sqrt(self.model_dim)

        flat_ids = shifted_ids[active_mask]
        flat_grads = dout[active_mask] * scale
        self.grads["embedding_table"].index_add_(
            0,
            flat_ids,
            flat_grads,
        )

        if self.pad_token_id is not None:
            self.grads["embedding_table"][self.pad_token_id] = 0

        return self.grads["embedding_table"]

    def step(self, learning_rate, weight_decay=0.0):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        grad = self.grads["embedding_table"]
        if weight_decay:
            grad = grad + weight_decay * self.embedding_table
        self.embedding_table -= learning_rate * grad

        if self.pad_token_id is not None:
            self.embedding_table[self.pad_token_id] = 0

    def to(self, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        self.device = device
        self.dtype = dtype

        self.embedding_table = self.embedding_table.to(
            device=device,
            dtype=dtype,
        )
        self.position_table = self.position_table.to(
            device=device,
            dtype=dtype,
        )
        self.grads["embedding_table"] = self.grads[
            "embedding_table"
        ].to(device=device, dtype=dtype)
        self.cache = None
        return self

    def state_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "model_dim": self.model_dim,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "max_sequence_length": self.max_sequence_length,
            "weights": {
                "embedding_table": self.embedding_table.detach().cpu(),
            },
        }

    def load_state_dict(self, state):
        if int(state["vocab_size"]) != self.vocab_size:
            raise ValueError("state vocab_size does not match this module")
        if int(state["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this module")
        if int(state["bos_token_id"]) != self.bos_token_id:
            raise ValueError("state bos_token_id does not match this module")
        if state["pad_token_id"] != self.pad_token_id:
            raise ValueError("state pad_token_id does not match this module")

        self.embedding_table = state["weights"]["embedding_table"].to(
            device=self.device,
            dtype=self.dtype,
        )
        if self.pad_token_id is not None:
            self.embedding_table[self.pad_token_id] = 0

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
            vocab_size=int(state["vocab_size"]),
            model_dim=int(state["model_dim"]),
            bos_token_id=int(state["bos_token_id"]),
            pad_token_id=state["pad_token_id"],
            max_sequence_length=int(state["max_sequence_length"]),
            device=device,
            dtype=dtype,
        )
        obj.load_state_dict(state)
        return obj


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ShiftedOutputEmbedder(
        vocab_size=16000,
        model_dim=256,
        bos_token_id=1,
        pad_token_id=0,
        max_sequence_length=128,
        device=device,
        dtype=torch.float32,
        seed=42,
    )

    targets = torch.tensor(
        [[20, 31, 2, 0], [42, 42, 2, 0]],
        device=device,
    )
    embeddings, shifted_ids = embedder(targets)
    embedder.backward(torch.ones_like(embeddings))

    print("Device:", embeddings.device)
    print("Shifted IDs:", shifted_ids)
    print("Embedding shape:", tuple(embeddings.shape))
