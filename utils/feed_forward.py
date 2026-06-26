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
            "PyTorch build before using FeedForward."
        )


class FeedForward:
    def __init__(
        self,
        model_dim,
        hidden_dim=None,
        multiple_of=64,
        device=None,
        dtype=None,
        seed=None,
    ):
        _require_torch()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if multiple_of <= 0:
            raise ValueError("multiple_of must be positive")

        self.model_dim = int(model_dim)
        self.multiple_of = int(multiple_of)
        self.hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else self._default_hidden_dim(self.model_dim, self.multiple_of)
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None

        self.W_gate = self._init_input_weight(generator)
        self.W_value = self._init_input_weight(generator)
        self.W_output = self._init_output_weight(generator)

        self.b_gate = self._zeros((self.hidden_dim,))
        self.b_value = self._zeros((self.hidden_dim,))
        self.b_output = self._zeros((self.model_dim,))

        self.cache = None
        self.zero_grad()

    @staticmethod
    def _default_hidden_dim(model_dim, multiple_of):
        estimated = (8 * model_dim) / 3
        return multiple_of * math.ceil(estimated / multiple_of)

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

    def _init_input_weight(self, generator):
        scale = 1.0 / math.sqrt(self.model_dim)
        return self._randn((self.model_dim, self.hidden_dim), generator) * scale

    def _init_output_weight(self, generator):
        scale = 1.0 / math.sqrt(self.hidden_dim)
        return self._randn((self.hidden_dim, self.model_dim), generator) * scale

    def parameters(self):
        return {
            "W_gate": self.W_gate,
            "W_value": self.W_value,
            "W_output": self.W_output,
            "b_gate": self.b_gate,
            "b_value": self.b_value,
            "b_output": self.b_output,
        }

    def zero_grad(self):
        self.grads = {
            "W_gate": self._zeros((self.model_dim, self.hidden_dim)),
            "W_value": self._zeros((self.model_dim, self.hidden_dim)),
            "W_output": self._zeros((self.hidden_dim, self.model_dim)),
            "b_gate": self._zeros((self.hidden_dim,)),
            "b_value": self._zeros((self.hidden_dim,)),
            "b_output": self._zeros((self.model_dim,)),
        }

    def _check_input(self, tensor, name):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension")
        if tensor.shape[-1] != self.model_dim:
            raise ValueError(
                f"{name} must have last dimension {self.model_dim}, "
                f"got {tensor.shape[-1]}"
            )

        return tensor.to(device=self.device, dtype=self.dtype).detach()

    def forward(self, x):
        x = self._check_input(x, "x")

        gate = x @ self.W_gate + self.b_gate
        sigmoid_gate = torch.sigmoid(gate)
        swish = gate * sigmoid_gate
        value = x @ self.W_value + self.b_value
        hidden = swish * value
        output = hidden @ self.W_output + self.b_output

        self.cache = {
            "x": x,
            "gate": gate,
            "sigmoid_gate": sigmoid_gate,
            "swish": swish,
            "value": value,
            "hidden": hidden,
        }
        return output

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        dout = self._check_input(dout, "dout")
        x = self.cache["x"]
        if dout.shape != x.shape:
            raise ValueError(
                f"dout shape must be {tuple(x.shape)}, got {tuple(dout.shape)}"
            )

        gate = self.cache["gate"]
        sigmoid_gate = self.cache["sigmoid_gate"]
        swish = self.cache["swish"]
        value = self.cache["value"]
        hidden = self.cache["hidden"]

        x_flat = x.reshape(-1, self.model_dim)
        dout_flat = dout.reshape(-1, self.model_dim)
        hidden_flat = hidden.reshape(-1, self.hidden_dim)

        self.grads["W_output"] += hidden_flat.transpose(0, 1) @ dout_flat
        self.grads["b_output"] += dout_flat.sum(dim=0)

        dhidden = dout @ self.W_output.transpose(0, 1)
        dswish = dhidden * value
        dvalue = dhidden * swish

        swish_derivative = sigmoid_gate + (
            gate * sigmoid_gate * (1.0 - sigmoid_gate)
        )
        dgate = dswish * swish_derivative

        dgate_flat = dgate.reshape(-1, self.hidden_dim)
        dvalue_flat = dvalue.reshape(-1, self.hidden_dim)

        self.grads["W_gate"] += x_flat.transpose(0, 1) @ dgate_flat
        self.grads["W_value"] += x_flat.transpose(0, 1) @ dvalue_flat
        self.grads["b_gate"] += dgate_flat.sum(dim=0)
        self.grads["b_value"] += dvalue_flat.sum(dim=0)

        dx = (
            dgate @ self.W_gate.transpose(0, 1)
            + dvalue @ self.W_value.transpose(0, 1)
        )
        return dx

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
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        for name in list(self.grads.keys()):
            self.grads[name] = self.grads[name].to(device=device, dtype=dtype)

        self.cache = None
        return self

    def state_dict(self):
        return {
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "multiple_of": self.multiple_of,
            "weights": {
                name: tensor.detach().cpu()
                for name, tensor in self.parameters().items()
            },
        }

    def load_state_dict(self, state):
        if int(state["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this module")
        if int(state["hidden_dim"]) != self.hidden_dim:
            raise ValueError("state hidden_dim does not match this module")

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
            hidden_dim=int(state["hidden_dim"]),
            multiple_of=int(state["multiple_of"]),
            device=device,
            dtype=dtype,
        )
        obj.load_state_dict(state)
        return obj


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ffn = FeedForward(
        model_dim=256,
        device=device,
        dtype=torch.float32,
        seed=42,
    )

    x = torch.randn(2, 16, 256, device=device)
    output = ffn(x)
    dx = ffn.backward(torch.ones_like(output))

    print("Device:", output.device)
    print("Hidden dimension:", ffn.hidden_dim)
    print("Output shape:", tuple(output.shape))
    print("Input gradient shape:", tuple(dx.shape))
