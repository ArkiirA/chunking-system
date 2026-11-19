# Chunking Module

A stdlib-only Python package that turns a Document and a ChunkingConfig into Chunks with metadata, adopted by the SanQua chatbots (tanyamesin, helpdesk, hotline) for RAG ingestion.

## Language

**Chunking Module**:
The shared pure library that splits a Document into Chunks according to a ChunkingConfig; owns no embeddings and no storage.
_Avoid_: chunking service, RAG pipeline

**Document**:
The module's input: an ordered list of Sections.
_Avoid_: file, PDF, article

**Section**:
A unit of text within a Document with optional structural metadata (page, section title, document title); a PDF page, a KB article, or a markdown header block.
_Avoid_: page, paragraph, block

**Chunk**:
A piece of Section text with metadata and an estimated token count; the unit a chatbot embeds and indexes.
_Avoid_: fragment, snippet, passage

**ChunkingConfig**:
The single configuration object controlling how a Document is chunked; carries per-collection and per-document overrides.
_Avoid_: chunking settings, splitter config

**Collection**:
A group of Documents that share one ChunkingConfig (a per-app KB, a per-manual-type corpus).
_Avoid_: index, database, bucket

**TokenEstimator**:
A pluggable interface that estimates the token count of a text; the word estimate is the default, a real tokenizer is an optional upgrade.
_Avoid_: tokenizer (that is one specific implementation)

**Word Estimate**:
The default token estimation: word count times a per-model factor (default 1.37, measured against the bge-m3 tokenizer on English technical text).
_Avoid_: approximation, guess

**Rechunking**:
When a chatbot changes a Collection's ChunkingConfig, all Chunks of that Collection are replaced: deleted, re-embedded, re-indexed.
_Avoid_: refresh, sync
