# Chunking module is a pure stdlib-only library

The chunking module adopted by the SanQua chatbots must not depend on any framework or third-party package. We decided it is a pure stdlib-only library — Document + ChunkingConfig in, Chunks with metadata out, no embeddings, no DB writes, no re-indexing — because the three chatbots each own their embedding service, DB schema, and replacement semantics; a shared module touching embeddings or storage would force a common schema on all of them.

**Considered options**: a LangChain/LlamaIndex-based splitter (ready-made, but pulls in a framework dependency and a tokenizer); a shared embedding + index service (convenient, but couples all chatbots to one schema).

**Consequences**: semantic and hierarchical strategies are deferred — semantic chunking needs per-sentence embeddings, which would break purity; a real tokenizer is an optional upgrade through TokenEstimator, not a default dependency.
