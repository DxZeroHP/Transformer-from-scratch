import ast
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.Tokenizer import UltTokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CLEAN_INPUT_PATH = os.path.join(DATA_DIR, "train_gen_phase1_clean.csv")
RAW_INPUT_PATH = os.path.join(DATA_DIR, "train_gen_phase1.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "tokenizer")

VOCAB_SIZE = 16000
MIN_FREQUENCY = 2
MAX_TRAINING_WORDS = 30000

# Set this to a number for a quick smoke test. Keep None for a real tokenizer.
ROW_LIMIT = None


def add_token(tokenizer, token):
    if token not in tokenizer.vocab:
        idx = len(tokenizer.vocab)
        tokenizer.vocab[token] = idx
        tokenizer.inverse_vocab[idx] = token


def ensure_required_tokens(tokenizer, texts):
    for token in tokenizer.special_tokens:
        add_token(tokenizer, token)

    add_token(tokenizer, tokenizer.end_of_word)
    for text in texts:
        for char in str(text):
            if char and not char.isspace():
                add_token(tokenizer, char)


def collect_from_clean_file(input_path):
    texts = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            training_text = row.get("training_text")
            if training_text:
                texts.append(training_text)
            if ROW_LIMIT and i >= ROW_LIMIT:
                break
    return texts


def collect_from_raw_file(input_path):
    texts = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            prompt = row.get("prompt")
            if prompt:
                texts.append(prompt)

            raw_messages = row.get("messages")
            if raw_messages:
                try:
                    parsed = ast.literal_eval(raw_messages)
                except Exception:
                    parsed = None

                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("content"):
                            texts.append(str(item["content"]))

            if ROW_LIMIT and i >= ROW_LIMIT:
                break
    return texts


def collect_texts():
    if os.path.exists(CLEAN_INPUT_PATH):
        return CLEAN_INPUT_PATH, collect_from_clean_file(CLEAN_INPUT_PATH)
    return RAW_INPUT_PATH, collect_from_raw_file(RAW_INPUT_PATH)


input_path, texts = collect_texts()
if not texts:
    raise RuntimeError(f"No tokenizer training text found in {input_path}")

print("Tokenizer input:", input_path)
print("Collected texts count:", len(texts))
print("Target merge operations:", VOCAB_SIZE)
print("Max unique words used for merge training:", MAX_TRAINING_WORDS)

tokenizer = UltTokenizer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    max_training_words=MAX_TRAINING_WORDS,
    vocab_file="vocab.txt",
    merges_file="merges.txt",
)

tokenizer.train(texts, show_progress=True)
ensure_required_tokens(tokenizer, texts)

os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save_files(OUTPUT_DIR)

print("Saved vocab and merges to", OUTPUT_DIR)
print("Final vocab size:", len(tokenizer.vocab))
print("Merge operations:", len(tokenizer.merges))

if len(tokenizer.merges) < VOCAB_SIZE:
    print(
        "Warning: merge training stopped before the target size. "
        "Lower min_frequency or provide more text if this is unexpected."
    )

examples = [
    "<bos> <user> Hello world! <assistant> Hi there. <eos>",
    "def tokenize(text): return text.split()",
]
for example in examples:
    ids = tokenizer.encode(example)
    print("Example text:", example)
    print("Example tokenize:", tokenizer.tokenize(example)[:30])
    print("Example encode:", ids[:30])
    print("Example decode:", tokenizer.decode(ids))
