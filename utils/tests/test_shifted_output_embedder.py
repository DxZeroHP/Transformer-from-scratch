import os
import tempfile
import unittest

import torch

from utils.shifted_output_embedder import ShiftedOutputEmbedder


class ShiftedOutputEmbedderTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.epsilon = 1e-5
        self.tolerance = 1e-7
        self.embedder = ShiftedOutputEmbedder(
            vocab_size=7,
            model_dim=3,
            bos_token_id=1,
            pad_token_id=0,
            max_sequence_length=8,
            device="cpu",
            dtype=self.dtype,
            seed=123,
        )
        self.targets = torch.tensor(
            [[2, 4, 2, 0], [5, 2, 0, 0]],
            dtype=torch.long,
        )

    def test_shift_right_inserts_bos_and_preserves_padding(self):
        shifted = self.embedder.shift_right(self.targets)
        expected = torch.tensor(
            [[1, 2, 4, 2], [1, 5, 2, 0]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(shifted, expected))

    def test_padding_output_and_gradient_are_zero(self):
        output, shifted = self.embedder.forward(self.targets)
        pad_positions = shifted == self.embedder.pad_token_id
        self.assertTrue(torch.equal(output[pad_positions], torch.zeros_like(
            output[pad_positions]
        )))

        self.embedder.zero_grad()
        self.embedder.forward(self.targets)
        self.embedder.backward(torch.ones_like(output))
        pad_grad = self.embedder.grads["embedding_table"][0]
        self.assertTrue(torch.equal(pad_grad, torch.zeros_like(pad_grad)))

    def test_repeated_token_gradients_accumulate(self):
        output, shifted = self.embedder.forward(self.targets)
        self.embedder.zero_grad()
        self.embedder.forward(self.targets)
        self.embedder.backward(torch.ones_like(output))

        token_id = 2
        occurrences = int((shifted == token_id).sum())
        expected = torch.full(
            (self.embedder.model_dim,),
            occurrences * self.embedder.model_dim ** 0.5,
            dtype=self.dtype,
        )
        actual = self.embedder.grads["embedding_table"][token_id]
        self.assertTrue(torch.allclose(actual, expected))

    def test_all_embedding_gradients_match_finite_differences(self):
        dout = torch.randn(
            2,
            4,
            self.embedder.model_dim,
            dtype=self.dtype,
        )

        self.embedder.zero_grad()
        self.embedder.forward(self.targets)
        analytical = self.embedder.backward(dout).clone()

        def loss():
            output, _ = self.embedder.forward(self.targets)
            return float((output * dout).sum())

        table = self.embedder.embedding_table
        for token_id in range(self.embedder.vocab_size):
            for feature in range(self.embedder.model_dim):
                original = table[token_id, feature].item()

                table[token_id, feature] = original + self.epsilon
                loss_plus = loss()
                table[token_id, feature] = original - self.epsilon
                loss_minus = loss()
                table[token_id, feature] = original

                numerical = (
                    loss_plus - loss_minus
                ) / (2.0 * self.epsilon)
                expected = analytical[token_id, feature].item()
                relative_error = abs(numerical - expected) / max(
                    1.0,
                    abs(numerical),
                    abs(expected),
                )
                self.assertLess(
                    relative_error,
                    self.tolerance,
                    (
                        f"embedding_table[{token_id}, {feature}] mismatch: "
                        f"numerical={numerical}, analytical={expected}, "
                        f"relative_error={relative_error}"
                    ),
                )

    def test_save_and_load_preserve_output(self):
        expected, expected_ids = self.embedder.forward(self.targets)
        path = os.path.join(
            tempfile.gettempdir(),
            "shifted_output_embedder_test.pt",
        )
        self.embedder.save(path)
        loaded = ShiftedOutputEmbedder.load(
            path,
            device="cpu",
            dtype=self.dtype,
        )
        actual, actual_ids = loaded.forward(self.targets)

        self.assertTrue(torch.equal(expected_ids, actual_ids))
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
