import os
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.decoder_only_transformer import DecoderOnlyTransformer


class DecoderOnlyTransformerTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.model = DecoderOnlyTransformer(
            vocab_size=16,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            max_sequence_length=8,
            bos_token_id=1,
            pad_token_id=0,
            ffn_hidden_dim=12,
            ffn_multiple_of=1,
            device="cpu",
            dtype=self.dtype,
            seed=123,
        )
        self.tokens = torch.tensor(
            [[4, 5, 6, 2, 0], [7, 8, 2, 0, 0]],
            dtype=torch.long,
        )

    def test_forward_loss_backward_and_generation_shapes(self):
        logits = self.model.forward(self.tokens)
        self.assertEqual(tuple(logits.shape), (2, 5, 16))
        self.assertTrue(torch.isfinite(logits).all())

        loss = self.model.loss_and_backward(self.tokens)
        self.assertTrue(torch.isfinite(loss))

        for gradient in self.model._all_gradient_tensors():
            self.assertTrue(torch.isfinite(gradient).all())

        generated = self.model.generate(torch.tensor([1, 4, 5]), max_new_tokens=3)
        self.assertEqual(tuple(generated.shape), (6,))

    def test_global_gradient_clipping_limits_total_norm(self):
        self.model.loss_and_backward(self.tokens)
        before = self.model.clip_grad_norm(max_norm=0.25)
        after_sq = torch.zeros((), dtype=self.dtype)
        for gradient in self.model._all_gradient_tensors():
            after_sq += (gradient * gradient).sum()
        after = torch.sqrt(after_sq)

        self.assertGreater(float(before), 0.0)
        self.assertLessEqual(float(after), 0.250001)

    def test_step_updates_parameters_and_preserves_pad_embedding(self):
        self.model.loss_and_backward(self.tokens)
        before = self.model.embedder.embedding_table.clone()
        self.model.step(learning_rate=1e-4, max_grad_norm=1.0)

        self.assertGreater(
            float((self.model.embedder.embedding_table - before).abs().sum()),
            0.0,
        )
        self.assertTrue(torch.equal(
            self.model.embedder.embedding_table[0],
            torch.zeros_like(self.model.embedder.embedding_table[0]),
        ))

    def test_one_small_step_does_not_increase_training_loss(self):
        initial_loss = self.model.loss_and_backward(self.tokens)
        self.model.step(learning_rate=1e-5, max_grad_norm=1.0)
        next_loss = self.model.loss_and_backward(self.tokens)

        self.assertLessEqual(float(next_loss), float(initial_loss) + 1e-8)

    def test_save_and_load_preserve_logits(self):
        expected = self.model.forward(self.tokens)
        path = os.path.join(
            tempfile.gettempdir(),
            "decoder_only_transformer_test.pt",
        )
        self.model.save(path)
        loaded = DecoderOnlyTransformer.load(
            path,
            device="cpu",
            dtype=self.dtype,
        )
        actual = loaded.forward(self.tokens)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
