import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from models.decoder_only_transformer import DecoderOnlyTransformer
from utils.Tokenizer import UltTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "decoder_chatbot.pth")

HOST = "127.0.0.1"
PORT = 8000

class ChatMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.turns = []

    def add_message(self, role, message):
        self.turns.append({"role": role, "message": message})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context(self):
        lines = []
        for turn in self.turns:
            prefix = "<user>" if turn["role"] == "user" else "<assistant>"
            lines.append(f"{prefix} {turn['message']}")
        return " ".join(lines)


class ChatHandler(BaseHTTPRequestHandler):
    tokenizer = UltTokenizer(
        vocab_file="vocab.txt",
        merges_file="merges.txt",
    )
    if not os.path.exists(os.path.join(TOKENIZER_DIR, "vocab.txt")):
        raise FileNotFoundError("Tokenizer files not found in data/tokenizer")
    tokenizer.load_files(TOKENIZER_DIR)

    model = DecoderOnlyTransformer.load(MODEL_PATH, device="cpu", dtype=torch.float32)
    memory = ChatMemory(max_turns=8)

    def _set_headers(self, status=200, content_type="text/html"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            self._set_headers(200, "text/html")
            with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "Not Found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        user_message = payload.get("message", "").strip()

        if not user_message:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Message is required"}).encode("utf-8"))
            return

        reply = self.handle_chat(user_message)
        self._set_headers(200, "application/json")
        self.wfile.write(json.dumps({"reply": reply}).encode("utf-8"))

    def handle_chat(self, user_message):
        self.memory.add_message("user", user_message)
        context = self.memory.get_context()

        prompt = f"<bos> <user> {context} <assistant>"
        prefix_ids = self.tokenizer.encode(prompt)
        if len(prefix_ids) == 0:
            return "Sorry, I couldn't encode the prompt."

        # Reserve tokens for the model output, ensuring we don't exceed model limit.
        max_output_tokens = min(50, self.model.max_sequence_length - len(prefix_ids))
        if max_output_tokens <= 0:
            return "Prompt is too long for the model."

        generated_ids = self.model.generate(prefix_ids, max_new_tokens=max_output_tokens, eos_token_id=self.tokenizer.vocab.get("<eos>"))
        response_ids = generated_ids[len(prefix_ids) :]
        response_text = self.tokenizer.decode(response_ids.tolist() if hasattr(response_ids, "tolist") else response_ids)
        self.memory.add_message("assistant", response_text)
        return response_text


def run_server():
    server = HTTPServer((HOST, PORT), ChatHandler)
    print(f"Serving chatbot at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
