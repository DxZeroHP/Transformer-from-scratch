try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for GPU tensor math. Install a CUDA-enabled "
            "PyTorch build before using ManualAdamW."
        )


def named_parameters_and_grads(module, prefix=""):
    seen = set()

    def visit(obj, obj_prefix):
        if obj is None:
            return

        if hasattr(obj, "parameters") and callable(obj.parameters) and hasattr(obj, "grads"):
            for name, parameter in obj.parameters().items():
                grad = obj.grads.get(name)
                if grad is None:
                    continue
                param_id = id(parameter)
                if param_id in seen:
                    continue
                seen.add(param_id)
                yield f"{obj_prefix}{name}", parameter, grad

        if hasattr(obj, "output_bias") and hasattr(obj, "grads"):
            grad = obj.grads.get("output_bias")
            if grad is not None:
                param_id = id(obj.output_bias)
                if param_id not in seen:
                    seen.add(param_id)
                    yield f"{obj_prefix}output_bias", obj.output_bias, grad

        if hasattr(obj, "embedder"):
            yield from visit(obj.embedder, f"{obj_prefix}embedder.")

        if hasattr(obj, "blocks"):
            for index, block in enumerate(obj.blocks):
                yield from visit(block, f"{obj_prefix}blocks.{index}.")

        for child_name in ("attention", "norm1", "feed_forward", "norm2"):
            if hasattr(obj, child_name):
                yield from visit(
                    getattr(obj, child_name),
                    f"{obj_prefix}{child_name}.",
                )

    yield from visit(module, prefix)


class ManualAdamW:
    def __init__(
        self,
        params,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=None,
        source_model=None,
    ):
        _require_torch()

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must satisfy 0 <= beta < 1")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer received no parameters")

        self.learning_rate = float(learning_rate)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self.timestep = 0
        self.state = {}
        self.source_model = source_model

        for name, parameter, _ in self.params:
            self.state[name] = {
                "m": torch.zeros_like(parameter),
                "v": torch.zeros_like(parameter),
            }

    @classmethod
    def from_model(cls, model, **kwargs):
        return cls(
            named_parameters_and_grads(model),
            source_model=model,
            **kwargs,
        )

    def refresh(self):
        if self.source_model is None:
            return

        self.params = list(named_parameters_and_grads(self.source_model))
        for name, parameter, _ in self.params:
            if name not in self.state:
                self.state[name] = {
                    "m": torch.zeros_like(parameter),
                    "v": torch.zeros_like(parameter),
                }
                continue

            if self.state[name]["m"].shape != parameter.shape:
                self.state[name]["m"] = torch.zeros_like(parameter)
                self.state[name]["v"] = torch.zeros_like(parameter)
            else:
                self.state[name]["m"] = self.state[name]["m"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                self.state[name]["v"] = self.state[name]["v"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )

    def zero_grad(self):
        self.refresh()
        for _, _, grad in self.params:
            grad.zero_()

    def grad_norm(self):
        self.refresh()
        total_sq_norm = None
        for _, _, grad in self.params:
            value = (grad * grad).sum()
            total_sq_norm = value if total_sq_norm is None else total_sq_norm + value
        return torch.sqrt(total_sq_norm)

    def clip_grad_norm(self, max_norm=None, epsilon=1e-12):
        self.refresh()
        max_norm = self.max_grad_norm if max_norm is None else max_norm
        if max_norm is None:
            return self.grad_norm()
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")

        norm = self.grad_norm()
        scale = torch.clamp(max_norm / (norm + epsilon), max=1.0)
        for _, _, grad in self.params:
            grad *= scale
        return norm

    def step(self):
        self.refresh()
        self.timestep += 1
        if self.max_grad_norm is not None:
            self.clip_grad_norm(self.max_grad_norm)

        bias_correction1 = 1.0 - self.beta1 ** self.timestep
        bias_correction2 = 1.0 - self.beta2 ** self.timestep

        for name, parameter, grad in self.params:
            state = self.state[name]
            state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (grad * grad)

            m_hat = state["m"] / bias_correction1
            v_hat = state["v"] / bias_correction2
            update = m_hat / (torch.sqrt(v_hat) + self.epsilon)

            if self.weight_decay:
                parameter -= self.learning_rate * self.weight_decay * parameter
            parameter -= self.learning_rate * update

    def state_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "timestep": self.timestep,
            "state": {
                name: {
                    "m": values["m"].detach().cpu(),
                    "v": values["v"].detach().cpu(),
                }
                for name, values in self.state.items()
            },
        }

    def load_state_dict(self, state):
        self.learning_rate = float(state["learning_rate"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.epsilon = float(state["epsilon"])
        self.weight_decay = float(state["weight_decay"])
        self.max_grad_norm = state["max_grad_norm"]
        self.timestep = int(state["timestep"])

        current = {name: parameter for name, parameter, _ in self.params}
        for name, values in state["state"].items():
            if name not in self.state:
                continue
            device = current[name].device
            dtype = current[name].dtype
            self.state[name]["m"] = values["m"].to(device=device, dtype=dtype)
            self.state[name]["v"] = values["v"].to(device=device, dtype=dtype)
