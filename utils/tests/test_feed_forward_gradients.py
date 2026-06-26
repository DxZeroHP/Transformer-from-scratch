import unittest

import torch

from utils.feed_forward import FeedForward


class FeedForwardGradientTest(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float64
        self.epsilon = 1e-5
        self.tolerance = 1e-7

        torch.manual_seed(123)
        self.ffn = FeedForward(
            model_dim=3,
            hidden_dim=4,
            multiple_of=1,
            device=self.device,
            dtype=self.dtype,
            seed=456,
        )
        self.x = torch.randn(1, 2, 3, device=self.device, dtype=self.dtype)
        self.dout = torch.randn(1, 2, 3, device=self.device, dtype=self.dtype)

    def _loss(self, x):
        return float((self.ffn.forward(x) * self.dout).sum())

    def _relative_error(self, numerical, analytical):
        return abs(numerical - analytical) / max(
            1.0,
            abs(numerical),
            abs(analytical),
        )

    def _assert_gradient_close(self, name, index, numerical, analytical):
        error = self._relative_error(numerical, analytical)
        self.assertLess(
            error,
            self.tolerance,
            (
                f"{name}{index} gradient mismatch: numerical={numerical}, "
                f"analytical={analytical}, relative_error={error}"
            ),
        )

    def _central_difference_input(self, index):
        x_plus = self.x.clone()
        x_minus = self.x.clone()
        x_plus[index] += self.epsilon
        x_minus[index] -= self.epsilon
        return (
            self._loss(x_plus) - self._loss(x_minus)
        ) / (2.0 * self.epsilon)

    def _central_difference_parameter(self, name, index):
        parameter = getattr(self.ffn, name)
        original = parameter[index].item()

        parameter[index] = original + self.epsilon
        loss_plus = self._loss(self.x)

        parameter[index] = original - self.epsilon
        loss_minus = self._loss(self.x)

        parameter[index] = original
        return (loss_plus - loss_minus) / (2.0 * self.epsilon)

    def test_all_manual_gradients_match_finite_differences(self):
        self.ffn.zero_grad()
        self.ffn.forward(self.x)
        dx = self.ffn.backward(self.dout).clone()
        parameter_grads = {
            name: gradient.clone()
            for name, gradient in self.ffn.grads.items()
        }

        for index in torch.cartesian_prod(
            *[torch.arange(size) for size in self.x.shape]
        ):
            index = tuple(int(value) for value in index)
            numerical = self._central_difference_input(index)
            analytical = dx[index].item()
            self._assert_gradient_close("x", index, numerical, analytical)

        for name, parameter in self.ffn.parameters().items():
            ranges = [torch.arange(size) for size in parameter.shape]
            for index in torch.cartesian_prod(*ranges):
                if index.ndim == 0:
                    index = (int(index),)
                else:
                    index = tuple(int(value) for value in index)

                numerical = self._central_difference_parameter(name, index)
                analytical = parameter_grads[name][index].item()
                self._assert_gradient_close(
                    name,
                    index,
                    numerical,
                    analytical,
                )


if __name__ == "__main__":
    unittest.main()
