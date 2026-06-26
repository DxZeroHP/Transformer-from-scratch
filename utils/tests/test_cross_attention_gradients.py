import os
import tempfile
import unittest

import torch

from utils.cross_attention import MultiHeadCrossAttention


class CrossAttentionGradientTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.epsilon = 1e-5
        self.tolerance = 1e-7
        self.attention = MultiHeadCrossAttention(
            model_dim=4,
            num_heads=2,
            max_query_length=4,
            max_context_length=4,
            device="cpu",
            dtype=self.dtype,
            seed=123,
        )
        torch.manual_seed(456)
        self.query = torch.randn(1, 2, 4, dtype=self.dtype)
        self.context = torch.randn(1, 3, 4, dtype=self.dtype)
        self.dout = torch.randn(1, 2, 4, dtype=self.dtype)

    def _loss(self, query, context, mask=None):
        output = self.attention.forward(
            query,
            context,
            attention_mask=mask,
        )
        return float((output * self.dout).sum())

    def _relative_error(self, numerical, analytical):
        return abs(numerical - analytical) / max(
            1.0,
            abs(numerical),
            abs(analytical),
        )

    def _assert_close(self, name, index, numerical, analytical):
        error = self._relative_error(numerical, analytical)
        self.assertLess(
            error,
            self.tolerance,
            (
                f"{name}{index} mismatch: numerical={numerical}, "
                f"analytical={analytical}, relative_error={error}"
            ),
        )

    def _all_indices(self, tensor):
        ranges = [torch.arange(size) for size in tensor.shape]
        for index in torch.cartesian_prod(*ranges):
            if index.ndim == 0:
                yield (int(index),)
            else:
                yield tuple(int(value) for value in index)

    def test_all_gradients_match_finite_differences(self):
        self.attention.zero_grad()
        self.attention.forward(self.query, self.context)
        dquery, dcontext = self.attention.backward(self.dout)
        parameter_grads = {
            name: gradient.clone()
            for name, gradient in self.attention.grads.items()
        }

        for name, source, analytical in (
            ("query", self.query, dquery),
            ("context", self.context, dcontext),
        ):
            for index in self._all_indices(source):
                plus = source.clone()
                minus = source.clone()
                plus[index] += self.epsilon
                minus[index] -= self.epsilon

                if name == "query":
                    loss_plus = self._loss(plus, self.context)
                    loss_minus = self._loss(minus, self.context)
                else:
                    loss_plus = self._loss(self.query, plus)
                    loss_minus = self._loss(self.query, minus)

                numerical = (
                    loss_plus - loss_minus
                ) / (2.0 * self.epsilon)
                self._assert_close(
                    name,
                    index,
                    numerical,
                    analytical[index].item(),
                )

        for name, parameter in self.attention.parameters().items():
            for index in self._all_indices(parameter):
                original = parameter[index].item()
                parameter[index] = original + self.epsilon
                loss_plus = self._loss(self.query, self.context)
                parameter[index] = original - self.epsilon
                loss_minus = self._loss(self.query, self.context)
                parameter[index] = original

                numerical = (
                    loss_plus - loss_minus
                ) / (2.0 * self.epsilon)
                self._assert_close(
                    name,
                    index,
                    numerical,
                    parameter_grads[name][index].item(),
                )

    def test_mask_blocks_context_positions(self):
        mask = torch.tensor([[[[True, True, False]]]])
        output = self.attention.forward(
            self.query,
            self.context,
            attention_mask=mask,
        )
        probabilities = self.attention.cache["probs"]

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(
            probabilities[..., 2],
            torch.zeros_like(probabilities[..., 2]),
        ))

        self.attention.zero_grad()
        self.attention.forward(
            self.query,
            self.context,
            attention_mask=mask,
        )
        _, dcontext = self.attention.backward(self.dout)
        self.assertTrue(torch.equal(
            dcontext[:, 2],
            torch.zeros_like(dcontext[:, 2]),
        ))

    def test_save_and_load_preserve_output(self):
        expected = self.attention.forward(
            self.query,
            self.context,
        )
        path = os.path.join(
            tempfile.gettempdir(),
            "cross_attention_test.pt",
        )
        self.attention.save(path)
        loaded = MultiHeadCrossAttention.load(
            path,
            device="cpu",
            dtype=self.dtype,
        )
        actual = loaded.forward(self.query, self.context)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
