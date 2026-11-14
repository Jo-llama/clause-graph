# 0003 Local embeddings (Ollama + bge-m3); LLM APIs only for public CUAD data

Status: Accepted, 2026-11-14 (step 5)

## Context
Step 5 adds semantic search: clause chunks are embedded and `assemble_context` uses the
best matches as entry points into the graph (ADR 0001). Contracts are confidential in any
real deployment. Under GDPR and typical NDAs, sending their text to a third-party
embedding API, especially one outside the EU, needs a legal basis and a processor
agreement. The 75 CUAD contracts are public, but the pipeline should behave the way
production would.

## Decision
- **Embeddings run locally**: Ollama (docker-compose service, CPU, port bound to
  127.0.0.1) with `bge-m3`, configured in `.env` (`EMBED_MODEL`, `OLLAMA_URL`).
  `src/clause_graph/embeddings.py` is the only module that talks to the model.
  bge-m3: multilingual (EU contracts are not only English), 1,024 dims, 8,192-token
  context, MIT licence. The vector index is Neo4j's native `chunk_embedding` (cosine).
- **LLM API calls** (harness, step 6) are a documented exception, allowed only for public
  CUAD data. Production would need an EU-hosted or local LLM.
- **Chunks**: up to 1,500 tokens, cut at sentence boundaries with up to 100 tokens of
  overlap. Near-empty chunks (bare headings) are skipped and counted. Each chunk keeps
  doc_id, section and character offsets into its clause, plus `text_hash`; an embedding is
  valid only while its `embed_hash` equals `text_hash`. embed_chunks.py re-embeds anything
  else, so a reload never leaves a stale vector.
- **Ollama options**: `truncate: false` (fail loudly instead of embedding a cut text) and
  `num_ctx = num_batch = 4096`. bge-m3 needs the whole input in one physical batch: with
  the default batch of 2,048, a 2,068-token chunk was rejected with HTTP 413.
- **chunker_version on every Chunk**:
  - **v1** sized chunks by estimated tokens (chars / 4). All 3,755 chunks embedded in step 5
    are v1.
  - **v2** sizes chunks by real bge-m3 tokens. It uses the model's Hugging Face
    tokenizer run locally (`tokenizers`, revision pinned in `embeddings.TOKENIZERS`).
    Only the public tokenizer file is downloaded; no text leaves the machine.
  - The loader keeps the version a contract already has, so a reload keeps its embeddings.
    New contracts, `--reset` and `--rechunk DOC_PREFIX` get v2.

## Measured (runs/embedding_metrics.json, i7-1165G7 4 cores, CPU only)
- Corpus: 75 contracts, 3,755 chunks, 956k real tokens; 0 failed chunks.
- Throughput: ~50-60 tokens/s (~740 chunks/h) with default settings; 87 tokens/s
  (1,296 chunks/h) with num_batch 4096. Threads, num_ctx and batch size beyond that
  made no measurable difference. Whole corpus ≈ 3 h.
- Scale: 10k contracts ≈ 500k chunks ≈ 127M tokens ≈ 400-660 h on CPU. Production needs
  a GPU (EU-hosted or on-prem) or overnight batching. Because embed_chunks.py works
  incrementally, each run only pays for new or changed chunks.
- Estimate vs real tokens: real tokens average 0.93-0.94 per estimated token over the
  corpus, so the estimate is fine on average. Per chunk it is not: p95 1.10, max 1.57 on
  dense tables and numbers. 23 v1 chunks exceed 1,500 real tokens, the largest at 2,068.
  The tokenizer count matches Ollama's prompt_eval_count exactly (2,068 for that chunk).
  All 23 are far below the 8,192 limit, so they are kept. Re-embedding would cost hours
  with no measured retrieval reason. A v2 dry run over the same contracts gives 3,740
  chunks, max 1,500 real tokens, and about 30 s of chunking time.

## Alternatives considered
- Hosted embedding API (OpenAI, Voyage, Cohere): faster and no ops work, but contract text
  leaves the machine. That is not acceptable for confidential contracts without an
  EU-hosted DPA.
- A smaller local model (e.g. nomic-embed-text, all-MiniLM): faster on CPU (not measured), but
  English-centric and shorter context. The model choice was kept, and the speed problem
  is handled by hardware and batching instead.
- sentence-transformers in-process: no extra service, but adds a torch dependency and
  model download to the app. Ollama keeps the model behind one HTTP call and the same
  setup works on a GPU host.
- A fixed safety margin on the estimate (e.g. cap at 1,500 / 1.6 estimated tokens) instead
  of v2: no new dependency, but the per-chunk ratio is unstable (0.9-1.57), so the margin
  would shrink most chunks by a third to protect a handful.
- An opt-in `--real-tokens` flag instead of chunker_version: same effect, but nothing in
  the graph would show which rule produced a chunk. chunker_version gives each chunk its
  provenance.

## Consequences
- No contract text leaves the machine for search. The cost is CPU time: about 3 h for the
  corpus, and production needs a GPU.
- The graph holds v1 and v2 chunks side by side. Chunk sizes differ slightly between
  contracts loaded before and after step 5; `chunker_version` and `tokens` make that
  visible, and `--rechunk` converges a contract when needed.
- Chunker v2 requires the `tokenizers` package and a one-time tokenizer download. The
  loader exits with a clear message if the tokenizer cannot be loaded.
- Changing EMBED_MODEL requires a matching entry in `embeddings.TOKENIZERS` and a full
  re-embed. The vector index dimension is checked against the model.
