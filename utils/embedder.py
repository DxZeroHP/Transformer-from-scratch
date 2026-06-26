import os

import numpy as np


class TokenEmbedder:
    def __init__(
        self,
        vocab_size,
        model_dim,
        pad_token_id=0,
        max_sequence_length=2048,
        use_positional_encoding=True,
        seed=None,
        dtype=np.float32,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.pad_token_id = pad_token_id
        self.max_sequence_length = int(max_sequence_length)
        self.use_positional_encoding = use_positional_encoding
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.model_dim)
        self.embedding_table = rng.normal(
            loc=0.0,
            scale=scale,
            size=(self.vocab_size, self.model_dim),
        ).astype(self.dtype)

        if self.pad_token_id is not None:
            self.embedding_table[self.pad_token_id] = 0

        self.position_table = self._build_sinusoidal_positions(
            self.max_sequence_length,
            self.model_dim,
            self.dtype,
        )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        model_dim,
        max_sequence_length=2048,
        use_positional_encoding=True,
        seed=None,
    ):
        pad_token_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        return cls(
            vocab_size=len(tokenizer.vocab),
            model_dim=model_dim,
            pad_token_id=pad_token_id,
            max_sequence_length=max_sequence_length,
            use_positional_encoding=use_positional_encoding,
            seed=seed,
        )

    def _build_sinusoidal_positions(self, max_sequence_length, model_dim, dtype):
        positions = np.arange(max_sequence_length, dtype=np.float32)[:, None]
        div_terms = np.exp(
            np.arange(0, model_dim, 2, dtype=np.float32)
            * (-np.log(10000.0) / model_dim)
        )

        table = np.zeros((max_sequence_length, model_dim), dtype=np.float32)
        table[:, 0::2] = np.sin(positions * div_terms)
        table[:, 1::2] = np.cos(positions * div_terms[: table[:, 1::2].shape[1]])
        return table.astype(dtype)

    def _as_batch(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim == 1:
            return token_ids[None, :], True
        if token_ids.ndim == 2:
            return token_ids, False
        raise ValueError("token_ids must be a 1D sequence or 2D batch")

    def _validate_token_ids(self, token_ids):
        if token_ids.size == 0:
            return

        min_id = int(token_ids.min())
        max_id = int(token_ids.max())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"token id out of range: expected 0 <= id < {self.vocab_size}, "
                f"got min={min_id}, max={max_id}"
            )

        if token_ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )

    def embed(self, token_ids):
        token_ids, squeeze_batch = self._as_batch(token_ids)
        self._validate_token_ids(token_ids)

        embeddings = self.embedding_table[token_ids]
        embeddings = embeddings * np.sqrt(self.model_dim).astype(self.dtype)

        if self.use_positional_encoding:
            sequence_length = token_ids.shape[1]
            embeddings = embeddings + self.position_table[:sequence_length]

        if self.pad_token_id is not None:
            pad_mask = token_ids == self.pad_token_id
            embeddings[pad_mask] = 0

        if squeeze_batch:
            return embeddings[0]
        return embeddings

    def __call__(self, token_ids):
        return self.embed(token_ids)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            embedding_table=self.embedding_table,
            position_table=self.position_table,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            pad_token_id=-1 if self.pad_token_id is None else self.pad_token_id,
            max_sequence_length=self.max_sequence_length,
            use_positional_encoding=self.use_positional_encoding,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        pad_token_id = int(data["pad_token_id"])
        obj = cls(
            vocab_size=int(data["vocab_size"]),
            model_dim=int(data["model_dim"]),
            pad_token_id=None if pad_token_id == -1 else pad_token_id,
            max_sequence_length=int(data["max_sequence_length"]),
            use_positional_encoding=bool(data["use_positional_encoding"]),
        )
        obj.embedding_table = data["embedding_table"]
        obj.position_table = data["position_table"]
        return obj


if __name__ == "__main__":
    embedder = TokenEmbedder(vocab_size=16000, model_dim=512, seed=42)
    token_ids = [1, 4, 250, 999, 2]
    embeddings = embedder(token_ids)
    print("Input shape:", np.asarray(token_ids).shape)
    print("Embedding shape:", embeddings.shape)
