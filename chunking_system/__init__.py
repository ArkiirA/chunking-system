"""chunking-system: pure stdlib-only chunking library for RAG ingestion.

Document + ChunkingConfig in, Chunks with metadata out. No embeddings,
no storage, no re-indexing (ADR-0001). The contract — types, config
surface, defaults, determinism, replacement semantics — is the durable
spec the future Rust port must match (ADR-0002).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

__all__ = [
    "Section", "Document", "Chunk", "ChunkingConfig",
    "TokenEstimator", "WordEstimator", "chunk", "chunk_report",
]


# --- types -----------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """A unit of text within a Document with optional structural metadata
    (page, section title, document title); a PDF page, a KB article, or a
    markdown header block."""
    text: str
    page: int | None = None
    section_title: str | None = None
    doc_title: str | None = None


@dataclass(frozen=True)
class Document:
    """The module's input: an ordered list of Sections."""
    sections: list[Section]


@dataclass(frozen=True)
class Chunk:
    """A piece of Section text with metadata and an estimated token count;
    the unit a chatbot embeds and indexes."""
    text: str
    embedded_text: str
    metadata: dict
    tokens: int


# --- token estimation -------------------------------------------------------

class TokenEstimator:
    """A pluggable interface that estimates the token count of a text; the
    word estimate is the default, a real tokenizer an optional upgrade."""

    def estimate(self, text: str) -> int:
        raise NotImplementedError


class WordEstimator(TokenEstimator):
    """Word count times a per-model factor (default 1.37, measured against
    the bge-m3 tokenizer on English technical text). Ceils, so an estimate
    never undercounts."""

    def __init__(self, factor: float = 1.37):
        self.factor = factor

    def estimate(self, text: str) -> int:
        return math.ceil(len(text.split()) * self.factor)


# --- config ------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """The single configuration object controlling how a Document is chunked;
    carries per-collection and per-document overrides. The dataclass defaults
    ARE the recommended (evidence-backed) values."""

    chunk_size_tokens: int = 512
    chunk_overlap_fraction: float = 0.25
    split_strategy: str = "recursive"
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", " ", ""])
    respect_structure: bool = True
    min_chunk_size_tokens: int = 128
    max_chunk_size_tokens: int = 2048
    embedding_model_context: int = 8192
    token_method: str = "word_estimate"
    words_to_tokens_factor: float = 1.37
    no_chunking_below_tokens: int = 512
    include_title: bool = True
    include_section: bool = True
    collection_overrides: dict = field(default_factory=dict)
    document_overrides: dict = field(default_factory=dict)

    STRATEGIES = ("fixed", "recursive", "sentence", "semantic", "hierarchical")
    TOKEN_METHODS = ("word_estimate", "tokenizer")

    # name -> (recommended, min, max); max None = derived (see variable_spec).
    # Deliberately un-annotated so the dataclass does not treat it as a field.
    _SPEC = {
        "chunk_size_tokens": (512, 256, 1024),
        "chunk_overlap_fraction": (0.25, 0.0, 0.25),
        "min_chunk_size_tokens": (128, 64, 256),
        "max_chunk_size_tokens": (2048, 128, None),
        "words_to_tokens_factor": (1.37, 1.0, 1.5),
        "no_chunking_below_tokens": (512, 0, 2048),
    }

    @classmethod
    def recommended(cls) -> "ChunkingConfig":
        """Factory for the evidence-backed defaults (research: 512 tokens,
        25% overlap, 128 min, 2048 guard, 1.37 factor, 512 no-chunking floor)."""
        return cls()

    def variable_spec(self, name: str) -> dict:
        """The recommended value and allowed range for one variable — the
        source of truth for the admin tab's current | recommended | range
        display and for the Rust port."""
        rec, lo, hi = self._SPEC[name]
        if hi is None:
            hi = int(0.9 * self.embedding_model_context)
        return {"recommended": rec, "min": lo, "max": hi}

    def validate(self) -> None:
        """Enforce the hard caps. Raises ValueError naming the variable."""
        if self.split_strategy not in self.STRATEGIES:
            raise ValueError(
                f"split_strategy {self.split_strategy!r} not in "
                f"{self.STRATEGIES}")
        if self.token_method not in self.TOKEN_METHODS:
            raise ValueError(
                f"token_method {self.token_method!r} not in "
                f"{self.TOKEN_METHODS}")
        for name in self._SPEC:
            rec, lo, hi = self._SPEC[name]
            value = getattr(self, name)
            if hi is None:
                hi = int(0.9 * self.embedding_model_context)
            if not (lo <= value <= hi):
                extra = (
                    " (90% of the embedding model context)"
                    if name == "max_chunk_size_tokens" else "")
                raise ValueError(
                    f"{name}={value} outside allowed range [{lo}, {hi}]{extra}")
        if self.min_chunk_size_tokens > self.chunk_size_tokens:
            raise ValueError(
                f"min_chunk_size_tokens={self.min_chunk_size_tokens} exceeds "
                f"chunk_size_tokens={self.chunk_size_tokens}")
        if self.chunk_overlap_fraction > 0.25:
            raise ValueError(
                f"chunk_overlap_fraction={self.chunk_overlap_fraction} above "
                "hard cap 0.25")
        if self.max_chunk_size_tokens > int(
                0.9 * self.embedding_model_context):
            raise ValueError(
                f"max_chunk_size_tokens={self.max_chunk_size_tokens} exceeds "
                f"90% of embedding model context "
                f"({self.embedding_model_context})")

    def _with_overrides(self, overrides: dict) -> "ChunkingConfig":
        """A validated copy of this config with named variables overridden.
        Raises ValueError for unknown variables or values that violate the
        hard caps (an override can never smuggle in an illegal config)."""
        eff = copy.copy(self)
        eff.separators = list(self.separators)
        eff.collection_overrides = dict(self.collection_overrides)
        eff.document_overrides = dict(self.document_overrides)
        for key, value in overrides.items():
            if key in ("collection_overrides", "document_overrides") \
                    or not hasattr(self, key):
                raise ValueError(f"unknown override variable {key!r}")
            setattr(eff, key, value)
        eff.validate()
        return eff

    def for_collection(self, name: str) -> "ChunkingConfig":
        """The effective config for one Collection (per-app KB, per-manual-
        type corpus); unknown collections get the base config."""
        return self._with_overrides(
            self.collection_overrides.get(name, {}))

    def for_document(self, name: str) -> "ChunkingConfig":
        """The effective config for one Document (the per-document escape
        hatch, e.g. a dense spec page that wants smaller chunks)."""
        return self._with_overrides(self.document_overrides.get(name, {}))


# --- entry point -------------------------------------------------------------

def _chunk_all(document: Document, config: ChunkingConfig,
               estimator: TokenEstimator | None = None
               ) -> tuple[list[Chunk], list[tuple[set, list[Chunk]]]]:
    """Shared core: validate, resolve the estimator, and split. Returns the
    flat chunk list plus per-section groups (word set, chunks) so
    chunk_report can check the boundary invariant."""
    config.validate()
    if config.split_strategy in ("semantic", "hierarchical", "sentence"):
        raise NotImplementedError(
            f"split_strategy {config.split_strategy!r} is not implemented in "
            "v1 (deferred: evidence does not show it beats recursive for the "
            "cost)")
    if config.token_method == "tokenizer" and estimator is None:
        raise ValueError(
            "token_method 'tokenizer' requires an explicit estimator")
    est = (estimator if estimator is not None
           else WordEstimator(config.words_to_tokens_factor))
    out: list[Chunk] = []
    groups: list[tuple[set, list[Chunk]]] = []
    if config.respect_structure:
        work: list[tuple[int, Section | None, str]] = [
            (i, s, s.text) for i, s in enumerate(document.sections)]
    else:
        # One text: section structure (page/section boundaries and
        # metadata) is ignored; paragraph separators inside the joined
        # text still guide the recursive cuts.
        work = [(0, None,
                 "\n\n".join(s.text for s in document.sections))]
    for section_index, section, text in work:
        base = {
            "source": section.doc_title if section else None,
            "page": section.page if section else None,
            "section": section.section_title if section else None,
            "title": section.doc_title if section else None,
            "section_index": section_index,
        }
        group: list[Chunk] = []
        for chunk_index, piece in enumerate(
                _split_section(text, config, est)):
            meta = dict(base)
            meta["chunk_index"] = chunk_index
            group.append(Chunk(
                text=piece,
                embedded_text=(
                    _embedded_text(piece, section, config)
                    if section is not None else piece),
                metadata=meta,
                tokens=est.estimate(piece),
            ))
        groups.append((set(text.split()), group))
        out.extend(group)
    return out, groups


def chunk(document: Document, config: ChunkingConfig,
          estimator: TokenEstimator | None = None) -> list[Chunk]:
    """Split a Document into Chunks according to a ChunkingConfig.

    Deterministic: the same Document + config always yields identical Chunks
    including chunk_index (the contract that makes Rechunking well-defined).
    Overlap never crosses section boundaries. token_method 'tokenizer'
    requires an explicit estimator (a real tokenizer); 'word_estimate' uses
    WordEstimator(config.words_to_tokens_factor).
    """
    out, _ = _chunk_all(document, config, estimator)
    return out


def chunk_report(document: Document, config: ChunkingConfig,
                 estimator: TokenEstimator | None = None) -> dict:
    """The tuning loop (adjust-inspect-commit): run the chunking and report
    the result without committing anything — the estimated chunk count, size
    distribution, and the boundary invariant check. The count is what makes
    the rechunk decision safe (prior-art: Dify's Confirm & Preview)."""
    out, groups = _chunk_all(document, config, estimator)
    sizes = [ch.tokens for ch in out]
    crossings = 0
    for wordset, group in groups:
        for ch in group:
            crossings += sum(1 for w in ch.text.split() if w not in wordset)
    return {
        "chunk_count": len(out),
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "avg": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        "total_tokens": sum(sizes),
        "boundary_crossings": crossings,
        "sizes": sizes,
    }


def _split_section(text: str, config: ChunkingConfig,
                   est: TokenEstimator) -> list[str]:
    """Split one section's text. Returns [] for empty text.

    Order of decisions: whole-section floor (no_chunking_below) -> fits-in-one
    check -> windowed split (fixed: fixed windows; recursive: separator-aware,
    later cycle) -> undersized-tail merge (settled decision: merge, not drop;
    the merge respects the max-context guard).
    """
    words = text.split()
    if not words:
        return []
    if (config.no_chunking_below_tokens > 0
            and est.estimate(text) <= config.no_chunking_below_tokens):
        return [text]
    if est.estimate(text) <= config.chunk_size_tokens:
        return [text]
    size = config.chunk_size_tokens
    overlap = int(size * config.chunk_overlap_fraction)
    if config.split_strategy == "recursive":
        return _recursive_split(text, words, config, est)
    # fixed: fixed windows (max words per window under the estimator)
    if isinstance(est, WordEstimator):
        k = max(1, int(size // est.factor))
    else:  # pluggable estimator: grow the window until the cap is hit
        k = 1
        while k < len(words) and est.estimate(
                " ".join(words[:k + 1])) <= size:
            k += 1
    pieces: list[str] = []
    start, n = 0, len(words)
    while start < n:
        end = min(start + k, n)
        pieces.append(" ".join(words[start:end]))
        if end == n:
            break
        start = end - overlap if overlap else end
    return _merge_undersized_tail(pieces, config, est)


def _merge_undersized_tail(pieces: list[str], config: ChunkingConfig,
                           est: TokenEstimator) -> list[str]:
    """Merge an undersized final piece into the previous one (settled
    decision: merge, not drop); the merge must respect the max-context
    guard, otherwise the tail stays separate."""
    if len(pieces) > 1 and est.estimate(pieces[-1]) < config.min_chunk_size_tokens:
        merged = pieces[-2] + " " + pieces[-1]
        if est.estimate(merged) <= config.max_chunk_size_tokens:
            pieces[-2] = merged
            pieces.pop()
    return pieces


def _recursive_split(text: str, words: list[str], config: ChunkingConfig,
                    est: TokenEstimator) -> list[str]:
    """Recursive (separator-hierarchy) split — the default strategy.

    Packs word windows up to the size cap, but prefers cut points at
    separator boundaries (paragraph, line, word — in config.separators
    order), keeping paragraphs together as long as possible. Overlap is
    word-level and never crosses section boundaries (each section is
    split independently).
    """
    n = len(words)
    size = config.chunk_size_tokens
    overlap = int(size * config.chunk_overlap_fraction)
    if isinstance(est, WordEstimator):
        k = max(1, int(size // est.factor))
    else:  # pluggable estimator: grow the window until the cap is hit
        k = 1
        while k < n and est.estimate(" ".join(words[:k + 1])) <= size:
            k += 1
    cuts = {n}
    for sep in config.separators:
        if not sep or sep == " ":
            continue
        for m in re.finditer(re.escape(sep), text):
            idx = len(text[:m.end()].split())
            if 0 < idx < n:
                cuts.add(idx)
    pieces: list[str] = []
    start = 0
    while start < n:
        candidates = [c for c in cuts if start < c <= start + k]
        end = max(candidates) if candidates else min(start + k, n)
        pieces.append(" ".join(words[start:end]))
        if end == n:
            break
        start = end - overlap if overlap else end
    return _merge_undersized_tail(pieces, config, est)


def _embedded_text(text: str, section: Section,
                   config: ChunkingConfig) -> str:
    """The text a chatbot embeds: title and/or section header prepended to
    the raw text (the contextual-prefix benefit, research §4.3)."""
    parts: list[str] = []
    if config.include_title and section.doc_title:
        parts.append(section.doc_title)
    if config.include_section and section.section_title:
        parts.append(section.section_title)
    return "\n".join(parts + [text]) if parts else text
