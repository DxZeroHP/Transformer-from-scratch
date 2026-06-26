import argparse
import collections
import csv
import math
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, Dataset

from models.decoder_only_transformer import DecoderOnlyTransformer
from utils.Tokenizer import UltTokenizer
from utils.optimizer import ManualAdamW

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
TRAIN_CSV = os.path.join(DATA_DIR, "train_gen_phase1_clean.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "decoder_chatbot.pth")

SPECIAL_USER = "<user>"
SPECIAL_ASSISTANT = "<assistant>"
SPECIAL_EOS = "<eos>"


def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder-only transformer on conversation data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum number of epochs. If omitted, training continues until validation F1 plateaus.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--val-csv", default=os.path.join(DATA_DIR, "test_gen_phase1_clean.csv"))
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--val-generate-tokens", type=int, default=64)
    parser.add_argument("--val-generate-examples", type=int, default=64,
        help="Number of validation examples used for generated response metrics; 0 disables generated text metrics.")
    parser.add_argument("--perplexity-threshold", type=float, default=20.0,
        help="Stop training when validation perplexity drops below this threshold.")
    parser.add_argument("--save-path", default=MODEL_SAVE_PATH)
    return parser.parse_args()


class PromptResponseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_sequence_length):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.examples = self._load_examples()

    def _load_examples(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Training CSV not found: {self.csv_path}")

        examples = []
        self.total_rows = 0
        self.skipped_rows = 0
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.total_rows += 1
                if "training_text" in row and row["training_text"]:
                    text = row["training_text"]
                    token_ids = self.tokenizer.encode(text)
                else:
                    prompt = row.get("prompt", "")
                    response = row.get("message") or ""
                    if not prompt or not response:
                        self.skipped_rows += 1
                        continue
                    text = f"{SPECIAL_USER} {prompt} {SPECIAL_ASSISTANT} {response} {SPECIAL_EOS}"
                    token_ids = self.tokenizer.encode(text)
                if len(token_ids) < 2:
                    self.skipped_rows += 1
                    continue
                if len(token_ids) > self.max_sequence_length:
                    token_ids = token_ids[-self.max_sequence_length:]
                examples.append(token_ids)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index], dtype=torch.long)


def collate_batch(batch, pad_token_id):
    max_len = max(item.size(0) for item in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for idx, item in enumerate(batch):
        padded[idx, : item.size(0)] = item
    return padded


def _trim_to_eos(token_ids, eos_id):
    token_ids = list(token_ids)
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]
    return token_ids


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


def _batch_examples(examples, batch_size):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptResponseEvalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_sequence_length):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.examples = self._load_examples()

    def _load_examples(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Validation CSV not found: {self.csv_path}")

        examples = []
        self.total_rows = 0
        self.skipped_rows = 0
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.total_rows += 1
                if "training_text" in row and row["training_text"]:
                    token_ids = self.tokenizer.encode(row["training_text"])
                    assistant_id = self.tokenizer.vocab.get(SPECIAL_ASSISTANT)
                    if assistant_id in token_ids:
                        idx = token_ids.index(assistant_id)
                        prompt_ids = token_ids[:idx + 1]
                        response_ids = token_ids[idx + 1:]
                    else:
                        self.skipped_rows += 1
                        continue
                else:
                    prompt = row.get("prompt", "")
                    response = row.get("message") or ""
                    if not prompt or not response:
                        self.skipped_rows += 1
                        continue
                    prompt_text = f"{SPECIAL_USER} {prompt} {SPECIAL_ASSISTANT}"
                    response_text = f"{response} {SPECIAL_EOS}"
                    prompt_ids = self.tokenizer.encode(prompt_text)
                    response_ids = self.tokenizer.encode(response_text)

                if len(prompt_ids) == 0 or len(response_ids) == 0:
                    self.skipped_rows += 1
                    continue

                if len(prompt_ids) + len(response_ids) > self.max_sequence_length:
                    allowed_response = self.max_sequence_length - len(prompt_ids)
                    if allowed_response <= 0:
                        self.skipped_rows += 1
                        continue
                    response_ids = response_ids[-allowed_response:]

                examples.append({
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "token_ids": torch.tensor(prompt_ids + response_ids, dtype=torch.long),
                })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def evaluate_dataset(model, tokenizer, dataset, batch_size, max_generate_tokens, generate_examples, device):
    if len(dataset) == 0:
        raise RuntimeError("No examples loaded from validation dataset.")

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

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    sampled_examples = []
    if generate_examples > 0:
        sampled_examples = dataset.examples[: min(generate_examples, len(dataset))]

    total_batches = len(eval_loader)
    original_train_flag = getattr(model, "train", None)
    model.train = False

    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader, start=1):
            batch_ids = collate_batch([item["token_ids"] for item in batch], pad_token_id).to(device)
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

            average_loss = total_nll / max(1, total_tokens)
            token_accuracy = total_correct / max(1, total_tokens)
            top5_accuracy = total_top5_correct / max(1, total_tokens)
            sequence_exact_match = exact_match_count / max(1, total_examples) if total_examples else 0.0
            average_f1 = total_f1 / max(1, total_examples) if total_examples else 0.0

            status = (
                f"Validation [{batch_index}/{total_batches}] - loss: {average_loss:.6f} "
                f"- token_acc: {token_accuracy:.4f} - top5_acc: {top5_accuracy:.4f} "
                f"- seq_em: {sequence_exact_match:.4f} - f1: {average_f1:.4f}"
            )
            sys.stdout.write(status + "\r")
            sys.stdout.flush()

        if sampled_examples:
            for example in sampled_examples:
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

    model.train = original_train_flag
    print()
    return {
        "perplexity": math.exp(min(100.0, total_nll / max(1, total_tokens))),
        "token_accuracy": total_correct / max(1, total_tokens),
        "top5_token_accuracy": total_top5_correct / max(1, total_tokens),
        "sequence_exact_match": exact_match_count / max(1, total_examples),
        "average_response_token_f1": total_f1 / max(1, total_examples),
        "average_generated_length": total_generated_length / max(1, total_examples),
        "average_target_length": total_target_length / max(1, total_examples),
        "num_examples": total_examples,
        "num_tokens": total_tokens,
    }


def build_model(tokenizer, args, device, dtype):
    return DecoderOnlyTransformer(
        vocab_size=len(tokenizer.vocab),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_sequence_length=args.max_sequence_length,
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        ffn_hidden_dim=args.ffn_hidden_dim,
        ffn_multiple_of=1,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )


def train():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    tokenizer = UltTokenizer(vocab_file="vocab.txt", merges_file="merges.txt")
    tokenizer.load_files(TOKENIZER_DIR)

    print(f"Using device: {device}")
    print(f"Tokenizer size: {len(tokenizer.vocab)}")

    dataset = PromptResponseDataset(args.train_csv, tokenizer, args.max_sequence_length)
    if len(dataset) == 0:
        raise RuntimeError(f"No training examples were loaded from the CSV: {args.train_csv}")

    print(
        f"Loaded {len(dataset)} examples from {args.train_csv} "
        f"(total rows={dataset.total_rows}, skipped={dataset.skipped_rows})"
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.vocab[tokenizer.pad_token]),
    )

    val_dataset = None
    if args.val_csv:
        val_dataset = PromptResponseEvalDataset(args.val_csv, tokenizer, args.max_sequence_length)
        print(f"Loaded {len(val_dataset)} validation examples from {args.val_csv} "
              f"(total rows={val_dataset.total_rows}, skipped={val_dataset.skipped_rows})")

    if os.path.exists(args.save_path):
        print(f"Found existing checkpoint at {args.save_path}; resuming training from saved model.")
        model = DecoderOnlyTransformer.load(args.save_path, device=device, dtype=dtype)
    else:
        model = build_model(tokenizer, args, device, dtype)

    optimizer = ManualAdamW.from_model(
        model,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    model.to(device=device, dtype=dtype)

    best_val_perplexity = float("inf")
    epoch = 0
    while True:
        epoch += 1
        if args.epochs is not None and epoch > args.epochs:
            print(f"Reached maximum epoch limit: {args.epochs}")
            break

        model.train = True
        epoch_loss = 0.0
        steps = 0
        total_batches = len(data_loader)

        for batch_index, batch_ids in enumerate(data_loader, start=1):
            batch_ids = batch_ids.to(device=device)
            loss = model.loss_and_backward(batch_ids)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.cpu())
            steps += 1

            avg_loss = epoch_loss / max(1, steps)
            progress = f"Epoch {epoch}/{args.epochs} [{batch_index}/{total_batches}]"
            status = f"{progress} - loss: {avg_loss:.6f}"
            sys.stdout.write(status + "\r")
            sys.stdout.flush()

        avg_loss = epoch_loss / max(1, steps)
        print(f"\nEpoch {epoch}/{args.epochs} - loss: {avg_loss:.6f} - steps: {steps}")

        if val_dataset is not None and len(val_dataset) > 0:
            print("Running validation...")
            val_metrics = evaluate_dataset(
                model,
                tokenizer,
                val_dataset,
                args.val_batch_size,
                args.val_generate_tokens,
                args.val_generate_examples,
                device,
            )
            val_perplexity = val_metrics["perplexity"]
            print(
                f"Validation results: perplexity={val_perplexity:.4f}, "
                f"token_acc={val_metrics['token_accuracy']:.4f}, "
                f"seq_em={val_metrics['sequence_exact_match']:.4f}, "
                f"f1={val_metrics['average_response_token_f1']:.4f}"
            )

            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                model.save(args.save_path)
                print(f"New best perplexity {val_perplexity:.4f}: saved checkpoint to {args.save_path}")

            if val_perplexity <= args.perplexity_threshold:
                print(f"Stopping training because validation perplexity {val_perplexity:.4f} <= threshold {args.perplexity_threshold:.4f}.")
                break
            else:
                print(f"Continuing training until validation perplexity is below {args.perplexity_threshold:.4f}.")

    if val_dataset is None or len(val_dataset) == 0:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        model.save(args.save_path)
        print(f"Saved model checkpoint to {args.save_path}")


if __name__ == "__main__":
    train()
