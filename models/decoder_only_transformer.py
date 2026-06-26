import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import torch
except ModuleNotFoundError:
    torch = None

from utils.add_and_normalization import AddAndNormalization
from utils.attention import MultiHeadSelfAttention
from utils.feed_forward import FeedForward
from utils.shifted_output_embedder import ShiftedOutputEmbedder


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for GPU tensor math. Install a CUDA-enabled "
            "PyTorch build before using DecoderOnlyTransformer."
        )


class DecoderBlock:
    def __init__(
        self,
        model_dim,
        num_heads,
        max_sequence_length,
        ffn_hidden_dim=None,
        ffn_multiple_of=64,
        device=None,
        dtype=None,
        seed=None,
    ):
        self.attention = MultiHeadSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + 1,
        )
        self.norm1 = AddAndNormalization(
            model_dim=model_dim,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = FeedForward(
            model_dim=model_dim,
            hidden_dim=ffn_hidden_dim,
            multiple_of=ffn_multiple_of,
            device=device,
            dtype=dtype,
            seed=None if seed is None else seed + 2,
        )
        self.norm2 = AddAndNormalization(
            model_dim=model_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, attention_mask=None):
        attention_output = self.attention.forward(
            x,
            causal=True,
            attention_mask=attention_mask,
        )
        x = self.norm1.forward(x, attention_output)
        feed_forward_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x, feed_forward_output)
        return x

    def backward(self, dout):
        d_norm2_residual, d_ffn_output = self.norm2.backward(dout)
        d_ffn_input = self.feed_forward.backward(d_ffn_output)
        d_after_norm1 = d_norm2_residual + d_ffn_input

        d_norm1_residual, d_attention_output = self.norm1.backward(d_after_norm1)
        d_attention_input = self.attention.backward(d_attention_output)
        return d_norm1_residual + d_attention_input

    def zero_grad(self):
        self.attention.zero_grad()
        self.norm1.zero_grad()
        self.feed_forward.zero_grad()
        self.norm2.zero_grad()

    def gradients(self):
        return {
            "attention": self.attention.grads,
            "norm1": self.norm1.grads,
            "feed_forward": self.feed_forward.grads,
            "norm2": self.norm2.grads,
        }

    def step(self, learning_rate, weight_decay=0.0):
        self.attention.step(learning_rate, weight_decay=weight_decay)
        self.norm1.step(learning_rate, weight_decay=weight_decay)
        self.feed_forward.step(learning_rate, weight_decay=weight_decay)
        self.norm2.step(learning_rate, weight_decay=weight_decay)

    def to(self, device=None, dtype=None):
        self.attention.to(device=device, dtype=dtype)
        self.norm1.to(device=device, dtype=dtype)
        self.feed_forward.to(device=device, dtype=dtype)
        self.norm2.to(device=device, dtype=dtype)
        return self

    def state_dict(self):
        return {
            "attention": self.attention.state_dict(),
            "norm1": self.norm1.state_dict(),
            "feed_forward": self.feed_forward.state_dict(),
            "norm2": self.norm2.state_dict(),
        }

    def load_state_dict(self, state):
        self.attention.load_state_dict(state["attention"])
        self.norm1.load_state_dict(state["norm1"])
        self.feed_forward.load_state_dict(state["feed_forward"])
        self.norm2.load_state_dict(state["norm2"])


class DecoderOnlyTransformer:
    def __init__(
        self,
        vocab_size,
        model_dim,
        num_heads,
        num_layers,
        max_sequence_length,
        bos_token_id,
        pad_token_id=0,
        ffn_hidden_dim=None,
        ffn_multiple_of=64,
        device=None,
        dtype=None,
        seed=None,
    ):
        _require_torch()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_sequence_length = int(max_sequence_length)
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_multiple_of = int(ffn_multiple_of)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        self.embedder = ShiftedOutputEmbedder(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            max_sequence_length=self.max_sequence_length,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )
        self.blocks = [
            DecoderBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                max_sequence_length=self.max_sequence_length,
                ffn_hidden_dim=self.ffn_hidden_dim,
                ffn_multiple_of=self.ffn_multiple_of,
                device=self.device,
                dtype=self.dtype,
                seed=None if seed is None else seed + 1000 * layer_index,
            )
            for layer_index in range(self.num_layers)
        ]
        self.output_bias = torch.zeros(
            self.vocab_size,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        self.grads = {
            "output_bias": torch.zeros_like(self.output_bias),
        }
        self.cache = None

    def to(self, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype

        self.device = device
        self.dtype = dtype

        self.embedder.to(device=device, dtype=dtype)
        for block in self.blocks:
            block.to(device=device, dtype=dtype)

        self.output_bias = self.output_bias.to(device=device, dtype=dtype)
        self.grads["output_bias"] = self.grads["output_bias"].to(device=device, dtype=dtype)
        self.cache = None
        return self

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        model_dim,
        num_heads,
        num_layers,
        max_sequence_length,
        ffn_hidden_dim=None,
        ffn_multiple_of=64,
        device=None,
        dtype=None,
        seed=None,
    ):
        return cls(
            vocab_size=len(tokenizer.vocab),
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            bos_token_id=tokenizer.vocab[tokenizer.bos_token],
            pad_token_id=tokenizer.vocab[tokenizer.pad_token],
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_multiple_of=ffn_multiple_of,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def _check_token_ids(self, token_ids, name):
        if not torch.is_tensor(token_ids):
            token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        token_ids = token_ids.to(device=self.device, dtype=torch.long).detach()
        squeeze_batch = False
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_batch = True
        elif token_ids.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D token IDs")

        if token_ids.shape[1] == 0:
            raise ValueError(f"{name} cannot have an empty sequence")
        if token_ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"{name} length {token_ids.shape[1]} exceeds "
                f"max_sequence_length {self.max_sequence_length}"
            )

        min_id = int(token_ids.min())
        max_id = int(token_ids.max())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"{name} token id out of range: expected 0 <= id < "
                f"{self.vocab_size}, got min={min_id}, max={max_id}"
            )
        return token_ids, squeeze_batch

    def _attention_mask_from_ids(self, token_ids):
        if self.pad_token_id is None:
            return None
        return (token_ids != self.pad_token_id)[:, None, None, :]

    def _embed_input_ids(self, input_ids):
        input_ids, squeeze_batch = self._check_token_ids(input_ids, "input_ids")
        sequence_length = input_ids.shape[1]
        embeddings = self.embedder.embedding_table[input_ids] * math.sqrt(
            self.model_dim
        )
        embeddings = embeddings + self.embedder.position_table[:sequence_length]
        if self.pad_token_id is not None:
            active_mask = input_ids != self.pad_token_id
            embeddings = embeddings.masked_fill(~active_mask.unsqueeze(-1), 0)
        if squeeze_batch:
            return embeddings[0], input_ids[0]
        return embeddings, input_ids

    def _forward_hidden(self, embeddings, attention_mask):
        x = embeddings
        for block in self.blocks:
            x = block.forward(x, attention_mask=attention_mask)
        return x

    def _project_logits(self, hidden):
        return hidden @ self.embedder.embedding_table.transpose(0, 1) + self.output_bias

    def forward(self, target_token_ids):
        embeddings, shifted_ids = self.embedder.forward(target_token_ids)
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
            shifted_ids = shifted_ids.unsqueeze(0)

        attention_mask = self._attention_mask_from_ids(shifted_ids)
        hidden = self._forward_hidden(embeddings, attention_mask)
        logits = self._project_logits(hidden)

        self.cache = {
            "hidden": hidden,
            "shifted_ids": shifted_ids,
            "logits": logits,
        }
        return logits

    def forward_input_ids(self, input_ids):
        embeddings, checked_ids = self._embed_input_ids(input_ids)
        squeeze_batch = False
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
            checked_ids = checked_ids.unsqueeze(0)
            squeeze_batch = True

        attention_mask = self._attention_mask_from_ids(checked_ids)
        hidden = self._forward_hidden(embeddings, attention_mask)
        logits = self._project_logits(hidden)
        if squeeze_batch:
            return logits[0]
        return logits

    def _cross_entropy_loss_and_gradient(self, logits, labels):
        labels, _ = self._check_token_ids(labels, "labels")
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape[:2])} does not match "
                f"labels shape {tuple(labels.shape)}"
            )

        if self.pad_token_id is None:
            active_mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            active_mask = labels != self.pad_token_id

        active_count = int(active_mask.sum())
        if active_count == 0:
            raise ValueError("labels contain no active non-padding tokens")

        shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(shifted_logits)
        probabilities = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

        flat_probabilities = probabilities.reshape(-1, self.vocab_size)
        flat_labels = labels.reshape(-1)
        flat_active_mask = active_mask.reshape(-1)
        active_rows = torch.nonzero(flat_active_mask, as_tuple=False).squeeze(1)
        target_probabilities = flat_probabilities[
            active_rows,
            flat_labels[active_rows],
        ].clamp_min(torch.finfo(logits.dtype).tiny)
        loss = -torch.log(target_probabilities).mean()

        dlogits = probabilities
        dlogits = dlogits.reshape(-1, self.vocab_size)
        dlogits[active_rows, flat_labels[active_rows]] -= 1.0
        dlogits[~flat_active_mask] = 0
        dlogits = dlogits.reshape_as(logits) / active_count
        return loss, dlogits

    def loss_and_backward(self, target_token_ids):
        self.zero_grad()
        logits = self.forward(target_token_ids)
        loss, dlogits = self._cross_entropy_loss_and_gradient(
            logits,
            target_token_ids,
        )
        self.backward(dlogits)
        return loss

    def backward(self, dlogits):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        dlogits = dlogits.to(device=self.device, dtype=self.dtype).detach()
        logits = self.cache["logits"]
        if dlogits.shape != logits.shape:
            raise ValueError(
                f"dlogits shape must be {tuple(logits.shape)}, "
                f"got {tuple(dlogits.shape)}"
            )

        hidden = self.cache["hidden"]
        hidden_flat = hidden.reshape(-1, self.model_dim)
        dlogits_flat = dlogits.reshape(-1, self.vocab_size)

        self.grads["output_bias"] += dlogits_flat.sum(dim=0)
        output_embedding_grad = dlogits_flat.transpose(0, 1) @ hidden_flat
        dhidden = dlogits @ self.embedder.embedding_table

        for block in reversed(self.blocks):
            dhidden = block.backward(dhidden)

        self.embedder.backward(dhidden)
        self.embedder.grads["embedding_table"] += output_embedding_grad
        if self.pad_token_id is not None:
            self.embedder.grads["embedding_table"][self.pad_token_id] = 0
        return self.embedder.grads["embedding_table"]

    def zero_grad(self):
        self.embedder.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.grads["output_bias"].zero_()

    def _all_gradient_tensors(self):
        tensors = [self.embedder.grads["embedding_table"], self.grads["output_bias"]]
        for block in self.blocks:
            for gradient_group in block.gradients().values():
                tensors.extend(gradient_group.values())
        return tensors

    def clip_grad_norm(self, max_norm, epsilon=1e-12):
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        total_sq_norm = torch.zeros((), device=self.device, dtype=self.dtype)
        for grad in self._all_gradient_tensors():
            total_sq_norm += (grad * grad).sum()
        total_norm = torch.sqrt(total_sq_norm)
        clip_scale = torch.clamp(max_norm / (total_norm + epsilon), max=1.0)
        for grad in self._all_gradient_tensors():
            grad *= clip_scale
        return total_norm

    def step(self, learning_rate, weight_decay=0.0, max_grad_norm=None):
        if max_grad_norm is not None:
            self.clip_grad_norm(max_grad_norm)
        self.embedder.step(learning_rate, weight_decay=weight_decay)
        for block in self.blocks:
            block.step(learning_rate, weight_decay=weight_decay)
        grad = self.grads["output_bias"]
        if weight_decay:
            grad = grad + weight_decay * self.output_bias
        self.output_bias -= learning_rate * grad

    def generate(self, input_ids, max_new_tokens, eos_token_id=None):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        generated, squeeze_batch = self._check_token_ids(input_ids, "input_ids")

        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.max_sequence_length:
                break
            logits = self.forward_input_ids(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        if squeeze_batch:
            return generated[0]
        return generated

    def state_dict(self):
        return {
            "config": {
                "vocab_size": self.vocab_size,
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_sequence_length": self.max_sequence_length,
                "bos_token_id": self.bos_token_id,
                "pad_token_id": self.pad_token_id,
                "ffn_hidden_dim": self.ffn_hidden_dim,
                "ffn_multiple_of": self.ffn_multiple_of,
            },
            "embedder": self.embedder.state_dict(),
            "blocks": [block.state_dict() for block in self.blocks],
            "output_bias": self.output_bias.detach().cpu(),
        }

    def load_state_dict(self, state):
        if int(state["config"]["vocab_size"]) != self.vocab_size:
            raise ValueError("state vocab_size does not match this model")
        if int(state["config"]["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this model")
        if int(state["config"]["num_layers"]) != self.num_layers:
            raise ValueError("state num_layers does not match this model")

        self.embedder.load_state_dict(state["embedder"])
        for block, block_state in zip(self.blocks, state["blocks"]):
            block.load_state_dict(block_state)
        self.output_bias = state["output_bias"].to(device=self.device, dtype=self.dtype)
        self.grads["output_bias"] = torch.zeros_like(self.output_bias)
        self.cache = None

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device=None, dtype=None):
        _require_torch()
        state = torch.load(path, map_location="cpu")
        config = dict(state["config"])
        model = cls(
            vocab_size=int(config["vocab_size"]),
            model_dim=int(config["model_dim"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            max_sequence_length=int(config["max_sequence_length"]),
            bos_token_id=int(config["bos_token_id"]),
            pad_token_id=config["pad_token_id"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            ffn_multiple_of=int(config["ffn_multiple_of"]),
            device=device,
            dtype=dtype,
        )
        model.load_state_dict(state)
        return model


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderOnlyTransformer(
        vocab_size=128,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        max_sequence_length=32,
        bos_token_id=1,
        pad_token_id=0,
        device=device,
        dtype=torch.float32,
        seed=42,
    )
    tokens = torch.tensor([[5, 9, 12, 2, 0, 0]], device=device)
    loss = model.loss_and_backward(tokens)
    model.step(learning_rate=1e-4, max_grad_norm=1.0)
    generated = model.generate(torch.tensor([1, 5, 9], device=device), 4)

    print("Device:", device)
    print("Loss:", float(loss.cpu()))
    print("Generated shape:", tuple(generated.shape))
