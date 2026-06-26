import os

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for GPU tensor math. Install a CUDA-enabled "
            "PyTorch build before using AddAndNormalization."
        )


class AddAndNormalization:
    def __init__(
        self,
        model_dim,
        epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        _require_torch()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.model_dim = int(model_dim)
        self.epsilon = float(epsilon)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        self.gamma = torch.ones(
            self.model_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        self.beta = self._zeros((self.model_dim,))

        self.cache = None
        self.zero_grad()

    def _zeros(self, shape):
        return torch.zeros(
            shape,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )

    def parameters(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
        }

    def zero_grad(self):
        self.grads = {
            "gamma": self._zeros((self.model_dim,)),
            "beta": self._zeros((self.model_dim,)),
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

        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return tensor.detach()

    def forward(self, residual, sublayer_output):
        residual = self._check_input(residual, "residual")
        sublayer_output = self._check_input(sublayer_output, "sublayer_output")

        if residual.shape != sublayer_output.shape:
            raise ValueError(
                "residual and sublayer_output must have identical shapes, "
                f"got {tuple(residual.shape)} and {tuple(sublayer_output.shape)}"
            )

        added = residual + sublayer_output
        mean = added.mean(dim=-1, keepdim=True)
        centered = added - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        inverse_std = torch.rsqrt(variance + self.epsilon)
        normalized = centered * inverse_std
        output = normalized * self.gamma + self.beta

        self.cache = {
            "normalized": normalized,
            "inverse_std": inverse_std,
            "gamma": self.gamma.clone(),
            "input_shape": residual.shape,
        }
        return output

    def __call__(self, residual, sublayer_output):
        return self.forward(residual, sublayer_output)

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")

        dout = self._check_input(dout, "dout")
        if dout.shape != self.cache["input_shape"]:
            raise ValueError(
                f"dout shape must be {tuple(self.cache['input_shape'])}, "
                f"got {tuple(dout.shape)}"
            )

        normalized = self.cache["normalized"]
        inverse_std = self.cache["inverse_std"]
        gamma = self.cache["gamma"]

        reduction_dims = tuple(range(dout.ndim - 1))
        self.grads["gamma"] += (dout * normalized).sum(dim=reduction_dims)
        self.grads["beta"] += dout.sum(dim=reduction_dims)

        dnormalized = dout * gamma
        feature_count = self.model_dim
        sum_dnormalized = dnormalized.sum(dim=-1, keepdim=True)
        sum_dnormalized_times_normalized = (
            dnormalized * normalized
        ).sum(dim=-1, keepdim=True)

        dadded = (
            inverse_std
            / feature_count
            * (
                feature_count * dnormalized
                - sum_dnormalized
                - normalized * sum_dnormalized_times_normalized
            )
        )

        return dadded, dadded.clone()

    def step(self, learning_rate, weight_decay=0.0, grad_clip=None):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

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

        self.cache = None
        return self

    def state_dict(self):
        return {
            "model_dim": self.model_dim,
            "epsilon": self.epsilon,
            "weights": {
                name: tensor.detach().cpu()
                for name, tensor in self.parameters().items()
            },
        }

    def load_state_dict(self, state):
        if int(state["model_dim"]) != self.model_dim:
            raise ValueError("state model_dim does not match this module")

        self.epsilon = float(state["epsilon"])
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
            epsilon=float(state["epsilon"]),
            device=device,
            dtype=dtype,
        )
        obj.load_state_dict(state)
        return obj


if __name__ == "__main__":
    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = AddAndNormalization(
        model_dim=256,
        device=device,
        dtype=torch.float32,
    )

    residual = torch.randn(2, 16, 256, device=device)
    sublayer_output = torch.randn(2, 16, 256, device=device)
    output = layer(residual, sublayer_output)
    d_residual, d_sublayer = layer.backward(torch.ones_like(output))

    print("Device:", output.device)
    print("Output shape:", tuple(output.shape))
    print("Residual gradient shape:", tuple(d_residual.shape))
    print("Sublayer gradient shape:", tuple(d_sublayer.shape))
