import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from utils.Tokenizer import UltTokenizer
from models.decoder_only_transformer import DecoderOnlyTransformer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "decoder_chatbot.pth")

print('PROJECT_ROOT', PROJECT_ROOT)
print('tokenizer files exist', os.path.exists(os.path.join(TOKENIZER_DIR, 'vocab.txt')), os.path.exists(os.path.join(TOKENIZER_DIR, 'merges.txt')))

tokenizer = UltTokenizer(vocab_file='vocab.txt', merges_file='merges.txt')
tokenizer.load_files(TOKENIZER_DIR)
print('vocab size', len(tokenizer.vocab))
print('special ids', {token: tokenizer.vocab.get(token) for token in ['<bos>', '<eos>', '<unk>', '<user>', '<assistant>']})

for text in ['Hello world', 'How are you?', 'Tell me a story']:
    ids = tokenizer.encode(text)
    print('text', repr(text), 'ids', ids, 'decode', tokenizer.decode(ids))

print('model exists', os.path.exists(MODEL_PATH))
if os.path.exists(MODEL_PATH):
    model = DecoderOnlyTransformer.load(MODEL_PATH, device='cpu', dtype=torch.float32)
    print('loaded model vocab_size', model.vocab_size, 'max_seq', model.max_sequence_length)
    prompt = '<bos> <user> Hello <assistant>'
    ids = tokenizer.encode(prompt)
    print('prompt ids', ids, 'decoded', tokenizer.decode(ids))
    generated = model.generate(ids, max_new_tokens=20, eos_token_id=tokenizer.vocab.get('<eos>'))
    print('generated ids', generated.tolist())
    print('generated text', tokenizer.decode(generated.tolist()))
