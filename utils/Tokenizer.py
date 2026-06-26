import os
import collections
import re

class UltTokenizer:
    def __init__(
        self,
        vocab_size=10000,
        min_frequency=2,
        vocab_file="vocab.txt",
        merges_file="merges.txt",
        max_training_words=8000,
    ):
        # Target size for vocabulary / merge operations.
        self.vocab_size = vocab_size
        # Minimum pair frequency before a merge is accepted.
        self.min_frequency = min_frequency
        # Keep only the most common unique words for BPE training.
        # This removes the noisy long tail that makes pure-Python BPE very slow.
        self.max_training_words = max_training_words
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # Special tokens used by the tokenizer and model.
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.user_token = "<user>"
        self.assistant_token = "<assistant>"
        # End-of-word marker used during tokenization.
        self.end_of_word = "</w>"

        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.user_token,
            self.assistant_token,
        ]
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self.merge_ranks = {}
        self._token_cache = {}

    def _print_progress(self, current, total, width=40):
        # Print a simple console progress bar for merge training.
        if total <= 0:
            return
        fraction = min(max(current / total, 0.0), 1.0)
        done = int(width * fraction)
        bar = "#" * done + "-" * (width - done)
        print(f"\rTraining tokenizer: [{bar}] {current}/{total} merges ({fraction * 100:.1f}%)", end="", flush=True)
        if current >= total:
            print()

    def _get_pair_frequencies(self, token_sequences):
        # Count adjacent token pairs, weighting each unique word by its frequency.
        pairs = collections.Counter()
        pairs_get = pairs.get
        for seq, count in token_sequences.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pairs[pair] = pairs_get(pair, 0) + count
        return pairs

    def _iter_pairs(self, seq):
        for i in range(len(seq) - 1):
            yield (seq[i], seq[i + 1])

    def _merge_pair(self, token_sequences, pair):
        # Replace every occurrence of the chosen pair with its merged form.
        merged_tokens = collections.Counter()
        first, second = pair
        merged = first + second

        for seq, count in token_sequences.items():
            if len(seq) < 2:
                merged_tokens[seq] += count
                continue

            i = 0
            new_seq = []
            changed = False
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == first and seq[i + 1] == second:
                    new_seq.append(merged)
                    i += 2
                    changed = True
                else:
                    new_seq.append(seq[i])
                    i += 1
            if changed:
                merged_tokens[tuple(new_seq)] += count
            else:
                merged_tokens[seq] += count

        return merged_tokens

    def _build_vocab(self, token_sequences):
        # Build the final vocabulary from all tokens present after BPE merges.
        tokens = set()
        for seq in token_sequences:
            tokens.update(seq)

        tokens = self.special_tokens + sorted(tokens - set(self.special_tokens))
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self._token_cache = {}

    def _word_to_tokens(self, word):
        # Convert a word into a list of characters plus the end-of-word marker.
        if word == "":
            return ()
        return tuple(word) + (self.end_of_word,)

    def _normalize_text(self, text):
        # Normalize whitespace and ensure the input is a string.
        if text is None:
            return ""
        text = str(text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _refresh_merge_ranks(self):
        # Rank lookup makes tokenization avoid scanning all merges repeatedly.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._token_cache = {}

    def _apply_bpe_to_word(self, word):
        cached = self._token_cache.get(word)
        if cached is not None:
            return cached

        pieces = self._word_to_tokens(word)
        if not self.merge_ranks or len(pieces) < 2:
            self._token_cache[word] = pieces
            return pieces

        while len(pieces) > 1:
            best_rank = None
            best_pair = None

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            merged = first + second
            new_pieces = []
            i = 0
            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == first and pieces[i + 1] == second:
                    new_pieces.append(merged)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = tuple(new_pieces)

        self._token_cache[word] = pieces
        return pieces

    def train(self, texts, show_progress=False):
        # Train the tokenizer from a list of raw text strings.
        token_sequences = collections.Counter()

        # Split on spaces and initialize each word as characters + </w>.
        for text in texts:
            text = self._normalize_text(text)
            for word in text.split(" "):
                if word:
                    if word in self.special_tokens:
                        token_sequences[(word,)] += 1
                    else:
                        token_sequences[self._word_to_tokens(word)] += 1

        if self.max_training_words and len(token_sequences) > self.max_training_words:
            token_sequences = collections.Counter(dict(token_sequences.most_common(self.max_training_words)))

        self.merges = []
        self._refresh_merge_ranks()
        while len(self.merges) < self.vocab_size:
            pair_freq = self._get_pair_frequencies(token_sequences)
            if not pair_freq:
                break
            best_pair, freq = max(pair_freq.items(), key=lambda item: item[1])
            if freq < self.min_frequency:
                break
            self.merges.append(best_pair)
            token_sequences = self._merge_pair(token_sequences, best_pair)
            progress_interval = max(1, self.vocab_size // 100)
            if show_progress and len(self.merges) % progress_interval == 0:
                self._print_progress(len(self.merges), self.vocab_size)

        # Print final progress if training ended early or completed.
        if show_progress:
            self._print_progress(len(self.merges), self.vocab_size)

        # After all merges, build vocabulary from the final token sequences.
        self._refresh_merge_ranks()
        self._build_vocab(token_sequences)

    def train_from_file(self, corpus_path, show_progress=False):
        # Train the tokenizer using text lines from a file.
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        texts = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)

        self.train(texts, show_progress=show_progress)

    def tokenize(self, text):
        # Convert a string into a list of subword tokens.
        text = self._normalize_text(text)
        tokens = []

        for word in text.split(" "):
            if not word:
                continue
            if word in self.special_tokens:
                tokens.append(word)
            else:
                pieces = self._apply_bpe_to_word(word)
                tokens.extend(pieces)
        return tokens

    def encode(self, text):
        # Convert text into integer token IDs.
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get(self.unk_token, 0)) for token in tokens]

    def decode(self, token_ids):
        # Convert token IDs back into decoded text.
        words = []
        current = ""

        for token_id in token_ids:
            token_id = int(token_id)
            token = self.inverse_vocab.get(token_id, self.unk_token)

            # If we hit a special token, flush the current word.
            if token in self.special_tokens:
                if current:
                    words.append(current)
                    current = ""
                words.append(token)
                continue

            # If the token ends a word, append the current word.
            if token.endswith(self.end_of_word):
                current += token[: -len(self.end_of_word)]
                words.append(current)
                current = ""
            else:
                current += token

        if current:
            words.append(current)

        return " ".join(words)

    def save(self, vocab_path=None, merges_path=None):
        # Save vocabulary and merge rules to disk.
        vocab_path = vocab_path or self.vocab_file
        merges_path = merges_path or self.merges_file

        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.vocab.items(), key=lambda item: item[1]):
                f.write(token + "\n")

        with open(merges_path, "w", encoding="utf-8") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

    def load(self, vocab_path=None, merges_path=None):
        # Load vocabulary and merge rules from disk.
        vocab_path = vocab_path or self.vocab_file
        merges_path = merges_path or self.merges_file

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"Merges file not found: {merges_path}")

        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    self.vocab[token] = idx

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        self.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))

        self._refresh_merge_ranks()

        # Ensure special tokens are present in the loaded vocabulary.
        for special in self.special_tokens:
            if special not in self.vocab:
                idx = len(self.vocab)
                self.vocab[special] = idx
                self.inverse_vocab[idx] = special

    def save_files(self, directory):
        # Save files into a directory using configured file names.
        os.makedirs(directory, exist_ok=True)
        self.save(os.path.join(directory, self.vocab_file), os.path.join(directory, self.merges_file))

    def load_files(self, directory):
        # Load files from a directory using configured file names.
        self.load(os.path.join(directory, self.vocab_file), os.path.join(directory, self.merges_file))
