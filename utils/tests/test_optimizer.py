import unittest

import torch

from models.decoder_only_transformer import DecoderOnlyTransformer
from utils.optimizer import ManualAdamW, named_parameters_and_grads


class ManualAdamWTest(unittest.TestCase):
    def test_first_step_matches_expected_adamw_update(self):
        parameter = torch.tensor([1.0, -2.0], dtype=torch.float64)
        grad = torch.tensor([0.5, -0.25], dtype=torch.float64)
        optimizer = ManualAdamW(
            [("p", parameter, grad)],
            learning_rate=0.1,
            weight_decay=0.01,
            max_grad_norm=None,
        )

        expected = parameter.clone()
        expected -= 0.1 * 0.01 * expected
        expected -= 0.1 * torch.sign(grad)

        optimizer.step()
        self.assertTrue(torch.allclose(parameter, expected))

    def test_global_gradient_clipping(self):
        parameter = torch.zeros(2, dtype=torch.float64)
        grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        optimizer = ManualAdamW(
            [("p", parameter, grad)],
            learning_rate=0.1,
            weight_decay=0.0,
            max_grad_norm=None,
        )

        before = optimizer.clip_grad_norm(1.0)
        after = optimizer.grad_norm()

        self.assertAlmostEqual(float(before), 5.0, places=7)
        self.assertAlmostEqual(float(after), 1.0, places=7)

    def test_optimizer_updates_decoder_model(self):
        model = DecoderOnlyTransformer(
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
            dtype=torch.float64,
            seed=123,
        )
        tokens = torch.tensor([[4, 5, 6, 2, 0]], dtype=torch.long)
        optimizer = ManualAdamW.from_model(
            model,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_grad_norm=1.0,
        )

        self.assertGreater(len(list(named_parameters_and_grads(model))), 0)
        loss = model.loss_and_backward(tokens)
        before = model.embedder.embedding_table.clone()
        norm = optimizer.grad_norm()
        optimizer.step()
        optimizer.zero_grad()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(norm), 0.0)
        self.assertGreater(
            float((model.embedder.embedding_table - before).abs().sum()),
            0.0,
        )
        for _, _, grad in optimizer.params:
            self.assertEqual(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
