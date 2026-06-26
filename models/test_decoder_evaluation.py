import collections
import csv
import math
import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.decoder_only_transformer import DecoderOnlyTransformer
from utils.Tokenizer import UltTokenizer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
TEST_CSV = os.path.join(DATA_DIR, "test_gen_phase1_clean.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "decoder_chatbot.pth")

SPECIAL_USER = "<user>"
SPECIAL_ASSISTANT = "<assistant>"
SPECIAL_EOS = "<eos>"


def _token_overlap_f1(pred_tokens, target_tokens):
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 0.0
    pred_counts = collections.Counter(pred_tokens)
    target_counts = collections.Counter(target_tokens)
    overlap = sum(min(pred_counts[token], target_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _trim_to_eos(token_ids, eos_id):
    token_ids = list(token_ids)
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]
    return token_ids


class PromptResponseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_sequence_length, max_examples=None):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_examples = max_examples
        self.examples = self._load_examples()

    def _load_examples(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Test CSV not found: {self.csv_path}")

        examples = []
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "training_text" in row and row["training_text"]:
                    token_ids = self.tokenizer.encode(row["training_text"])
                    assistant_id = self.tokenizer.vocab.get(SPECIAL_ASSISTANT)
                    if assistant_id in token_ids:
                        idx = token_ids.index(assistant_id)
                        prompt_ids = token_ids[:idx + 1]
                        response_ids = token_ids[idx + 1:]
                        full_ids = token_ids
                    else:
                        continue
                else:
                    prompt = row.get("prompt", "")
                    response = row.get("message") or ""
                    if not prompt or not response:
                        continue
                    prompt_text = f"{SPECIAL_USER} {prompt} {SPECIAL_ASSISTANT}"
                    response_text = f"{response} {SPECIAL_EOS}"
                    prompt_ids = self.tokenizer.encode(prompt_text)
                    response_ids = self.tokenizer.encode(response_text)
                    full_ids = prompt_ids + response_ids

                if len(prompt_ids) + len(response_ids) > self.max_sequence_length:
                    continue

                examples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "token_ids": torch.tensor(full_ids, dtype=torch.long),
                    }
                )
                if self.max_examples is not None and len(examples) >= self.max_examples:
                    break

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _batch_examples(examples, batch_size):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def evaluate_model(model, tokenizer, dataset, batch_size=16, max_generate_tokens=64, device="cpu"):
    model.to(device=device, dtype=torch.float32)
    pad_token_id = tokenizer.vocab[tokenizer.pad_token]
    eos_token_id = tokenizer.vocab[SPECIAL_EOS]

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_top5_correct = 0
    total_examples = 0
    exact_match_count = 0
    total_f1 = 0.0
    total_generated_length = 0
    total_target_length = 0

    total_batches = math.ceil(len(dataset.examples) / batch_size)
    for batch_index, batch in enumerate(_batch_examples(dataset.examples, batch_size), start=1):
        batch_ids = torch.nn.utils.rnn.pad_sequence(
            [example["token_ids"] for example in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)

        logits = model.forward(batch_ids)
        labels = batch_ids
        active_mask = labels != pad_token_id

        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        nll = -selected_log_probs

        total_nll += float(nll[active_mask].sum().cpu())
        total_tokens += int(active_mask.sum().cpu())

        predictions = logits.argmax(dim=-1)
        total_correct += int((predictions == labels).masked_select(active_mask).sum().cpu())

        if logits.shape[-1] >= 5:
            top5 = logits.topk(5, dim=-1).indices
            label_matches = top5.eq(labels.unsqueeze(-1)).any(dim=-1)
            total_top5_correct += int(label_matches.masked_select(active_mask).sum().cpu())

        for example, prediction in zip(batch, predictions.cpu().tolist()):
            prompt_ids = example["prompt_ids"]
            target_ids = example["response_ids"]
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=min(max_generate_tokens, len(target_ids) + 10),
                eos_token_id=eos_token_id,
            )
            generated_response = generated_ids[len(prompt_ids) :].tolist()
            generated_response = _trim_to_eos(generated_response, eos_token_id)
            target_response = _trim_to_eos(target_ids[:-1], eos_token_id)

            if generated_response == target_response:
                exact_match_count += 1

            total_f1 += _token_overlap_f1(generated_response, target_response)
            total_generated_length += len(generated_response)
            total_target_length += len(target_response)
            total_examples += 1

        average_loss = total_nll / max(1, total_tokens)
        token_accuracy = total_correct / max(1, total_tokens)
        top5_accuracy = total_top5_correct / max(1, total_tokens)
        sequence_exact_match = exact_match_count / max(1, total_examples) if total_examples else 0.0
        average_f1 = total_f1 / max(1, total_examples) if total_examples else 0.0
        progress = f"Batch {batch_index}/{total_batches}"
        status = (
            f"{progress} - loss: {average_loss:.6f} - token_acc: {token_accuracy:.4f} "
            f"- top5_acc: {top5_accuracy:.4f} - seq_em: {sequence_exact_match:.4f} "
            f"- avg_f1: {average_f1:.4f}"
        )
        sys.stdout.write(status + "\r")
        sys.stdout.flush()

    print()
    average_loss = total_nll / max(1, total_tokens)
    metrics = {
        "perplexity": math.exp(min(100.0, average_loss)),
        "token_accuracy": total_correct / max(1, total_tokens),
        "top5_token_accuracy": total_top5_correct / max(1, total_tokens),
        "sequence_exact_match": exact_match_count / max(1, total_examples),
        "average_response_token_f1": total_f1 / max(1, total_examples),
        "average_generated_length": total_generated_length / max(1, total_examples),
        "average_target_length": total_target_length / max(1, total_examples),
        "num_examples": total_examples,
        "num_tokens": total_tokens,
    }
    return metrics


class DecoderEvaluationTest(unittest.TestCase):
    def test_evaluate_decoder_model_on_test_csv(self):
        if not os.path.exists(MODEL_PATH):
            self.skipTest(f"Model checkpoint not found: {MODEL_PATH}")

        tokenizer = UltTokenizer(vocab_file="vocab.txt", merges_file="merges.txt")
        tokenizer.load_files(TOKENIZER_DIR)
        self.assertIn(SPECIAL_EOS, tokenizer.vocab)

        dataset = PromptResponseDataset(
            TEST_CSV,
            tokenizer,
            max_sequence_length=256,
        )
        self.assertGreater(len(dataset), 0, "No test examples were loaded from the test CSV.")

        model = DecoderOnlyTransformer.load(MODEL_PATH, device="cpu", dtype=torch.float32)
        metrics = evaluate_model(model, tokenizer, dataset, batch_size=8, max_generate_tokens=64, device="cpu")

        self.assertIn("perplexity", metrics)
        self.assertTrue(math.isfinite(metrics["perplexity"]))
        self.assertGreater(metrics["num_examples"], 0)
        self.assertGreater(metrics["num_tokens"], 0)
        self.assertGreaterEqual(metrics["token_accuracy"], 0.0)
        self.assertLessEqual(metrics["token_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["top5_token_accuracy"], 0.0)
        self.assertLessEqual(metrics["top5_token_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["sequence_exact_match"], 0.0)
        self.assertLessEqual(metrics["sequence_exact_match"], 1.0)
        self.assertGreaterEqual(metrics["average_response_token_f1"], 0.0)
        self.assertLessEqual(metrics["average_response_token_f1"], 1.0)

        print("\nEvaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    unittest.main()
