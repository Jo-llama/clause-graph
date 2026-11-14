"""The only module that talks to the embedding model.

Embeddings run locally in Ollama (docker-compose service "ollama"), so contract
text never leaves this machine. Model and endpoint come from .env:
  EMBED_MODEL=bge-m3
  OLLAMA_URL=http://localhost:11434
Swapping the model is a config change plus re-running scripts/embed_chunks.py
(the vector index dimension is taken from the model's output).

count_tokens() counts the model's real tokens with its Hugging Face tokenizer, run
locally: only the public tokenizer file is downloaded (once, pinned revision), no
text is sent anywhere. Chunker v2 uses it to size chunks (ADR 0003).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

from clause_graph.chunking import collapse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "bge-m3"
DEFAULT_URL = "http://localhost:11434"
# Per input. v1 chunks are capped at 1,500 *estimated* tokens, but dense text (numbers,
# tables) runs up to 1.57x the estimate (max 2,068 real); bge-m3 needs the whole input in
# one physical batch, so num_batch must be as large as the context.
NUM_CTX = 4096
TIMEOUT_S = 600
# Ollama model -> (Hugging Face repo, pinned revision) of the same tokenizer.
# Checked: BONTON...#H6#2 is 2,068 tokens here and in Ollama's prompt_eval_count.
TOKENIZERS = {"bge-m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")}


class EmbeddingError(Exception):
    """The embedding model is unreachable or rejected the input."""


class EmbeddingInputError(EmbeddingError):
    """Ollama rejected this input (e.g. too long); other inputs may still work."""


@dataclass(frozen=True)
class Config:
    model: str
    url: str


@dataclass(frozen=True)
class Batch:
    vectors: list[list[float]]
    tokens: int  # tokens the model actually processed (prompt_eval_count)


def config() -> Config:
    load_dotenv(ROOT / ".env")
    return Config(
        model=os.environ.get("EMBED_MODEL", DEFAULT_MODEL),
        url=os.environ.get("OLLAMA_URL", DEFAULT_URL).rstrip("/"),
    )


def embed(texts: list[str], cfg: Config | None = None) -> Batch:
    """Embed texts (whitespace-collapsed) in one request to Ollama /api/embed."""
    cfg = cfg or config()
    payload = {
        "model": cfg.model,
        "input": [collapse(t) for t in texts],
        "truncate": False,  # fail loudly instead of silently embedding a cut text
        "options": {"num_ctx": NUM_CTX, "num_batch": NUM_CTX},
    }
    request = urllib.request.Request(
        f"{cfg.url}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code == 404 or "not found" in detail:
            raise EmbeddingError(
                f"Embedding model '{cfg.model}' is not available in Ollama. "
                f"Pull it with: docker exec clause-graph-ollama ollama pull {cfg.model}"
            ) from e
        if 400 <= e.code < 500:
            raise EmbeddingInputError(f"Ollama rejected the input ({e.code}): {detail}") from e
        raise EmbeddingError(f"Ollama failed ({e.code}): {detail}") from e
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise EmbeddingError(
            f"Ollama is not reachable at {cfg.url}. Start it with: docker compose up -d ollama"
        ) from e
    vectors = body.get("embeddings") or []
    if len(vectors) != len(texts):
        raise EmbeddingError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts.")
    return Batch(vectors=vectors, tokens=int(body.get("prompt_eval_count", 0)))


def embed_query(text: str, cfg: Config | None = None) -> list[float]:
    if not collapse(text):
        raise EmbeddingError("query is empty.")
    return embed([text], cfg).vectors[0]


@cache
def _tokenizer(model: str):
    from tokenizers import Tokenizer

    if model not in TOKENIZERS:
        raise EmbeddingError(
            f"No tokenizer configured for '{model}'; add it to embeddings.TOKENIZERS."
        )
    repo, revision = TOKENIZERS[model]
    try:
        return Tokenizer.from_pretrained(repo, revision=revision)
    except Exception as e:  # network or cache failure
        raise EmbeddingError(f"Could not load the {repo} tokenizer: {e}") from e


def count_tokens(text: str, cfg: Config | None = None) -> int:
    """Real tokens the model sees for this text, incl. special tokens (= prompt_eval_count)."""
    cfg = cfg or config()
    return len(_tokenizer(cfg.model).encode(collapse(text)).ids)
