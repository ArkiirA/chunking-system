# chunking-system

Pure stdlib-only chunking library for RAG ingestion. **Document + ChunkingConfig in, Chunks with metadata out.** No embeddings, no storage, no re-indexing — the chatbots own those (ADR-0001).

The chatbots are being ported to Rust; **the contract of this module — the types, the config surface, the defaults, the determinism and replacement semantics — is the durable specification the Rust implementation must match** (ADR-0002). The Python package is the current implementation of that spec.

## Quickstart

```python
from chunking_system import ChunkingConfig, Document, Section, chunk, chunk_report

doc = Document(sections=[
    Section(text=page_text, page=12, section_title="Hydraulic System",
            doc_title="HyCap 160"),
])

config = ChunkingConfig.recommended()   # evidence-backed defaults
chunks = chunk(doc, config)             # deterministic
report = chunk_report(doc, config)      # inspect before committing
```

Each `Chunk` carries `text` (raw), `embedded_text` (title/section header
prepended — the contextual-prefix benefit), `metadata`
(`source`, `page`, `section`, `title`, `section_index`, `chunk_index`) and
an estimated `tokens` count.

## The contract

- **Determinism**: the same Document + config always yields identical Chunks
  including `chunk_index`. This is what makes **Rechunking** well-defined:
  when a chatbot changes a Collection's config, all of that Collection's
  Chunks are replaced (deleted, re-embedded, re-indexed).
- **Config surface**: one `ChunkingConfig` per Collection, with
  per-collection and per-document overrides
  (`config.for_collection(name)`, `config.for_document(name)` — overrides are
  validated and can never smuggle in an illegal config). `validate()`
  enforces the hard caps (overlap ≤ 0.25, max chunk ≤ 90% of the embedding
  model's context, min ≤ size, size/factor/floor ranges).
- **Recommended values + ranges** are exposed per variable via
  `config.variable_spec(name)` — the single source of truth for the admin
  tab's *current | recommended | range* display and for the Rust port.
- **Strategies (v1)**: `recursive` (default — separator-preferred cuts,
  paragraphs kept together) and `fixed`. `semantic` / `hierarchical` /
  `sentence` are reserved enum values that raise `NotImplementedError`
  (deferred: the evidence does not show they beat recursive for the cost;
  parent-child is the known high-value gap — prior-art research).
- **Token estimation**: `WordEstimator` (word count × per-model factor,
  default 1.37 — measured against the bge-m3 tokenizer) by default, per the
  on-prem no-tokenizer-dependency constraint. A real tokenizer is an
  optional upgrade: pass any `TokenEstimator` to `chunk()` /
  `chunk_report()`; `token_method = "tokenizer"` requires one.
- **Boundaries**: sections are the structural unit (PDF page, KB article,
  markdown header block). Overlap never crosses section boundaries;
  `chunk_report` checks the invariant (`boundary_crossings`).
- **Undersized tails are merged** into the previous chunk (never dropped),
  subject to the max-context guard.

## Defaults (evidence-backed, see `docs/research` in the SanQua workspace)

| variable | recommended | range |
|---|---|---|
| `chunk_size_tokens` | 512 | 256–1024 |
| `chunk_overlap_fraction` | 0.25 | 0.0–0.25 (hard cap) |
| `min_chunk_size_tokens` | 128 | 64–256 |
| `max_chunk_size_tokens` | 2048 | ≤ 90% of `embedding_model_context` |
| `words_to_tokens_factor` | 1.37 | 1.0–1.5 |
| `no_chunking_below_tokens` | 512 | 0–2048 |

## Dependencies

None at runtime (stdlib only). `pytest` is a dev-only dependency.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Layout

- `GLOSSARY.md` — the module's domain language
- `docs/adr/` — ADR-0001 (pure stdlib-only), ADR-0002 (Rust port / contract is durable)
