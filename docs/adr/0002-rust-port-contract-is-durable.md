# The chatbots are being ported to Rust; the module contract is the durable spec

The user stated the SanQua chatbots will be converted to Rust after being developed in Python. We therefore treat the chunking module's contract — the Document/Section/Chunk shapes, the ChunkingConfig surface, its defaults, and the determinism + replacement semantics — as the durable specification the future Rust implementation must match, with the Python package as the current implementation of that spec.

**Consequences**: the module stays pure and stdlib-only (no framework coupling to carry into the port); the config surface and defaults are documented and versioned independently of the Python code; the dependency mechanism (sibling clone + path dependency, Q16) is a Python-era convenience, not a long-term commitment.
