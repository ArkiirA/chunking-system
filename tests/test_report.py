"""Cycle 4: chunk_report (tuning loop), overrides, pluggable estimator."""
from chunking_system import (
    ChunkingConfig,
    Document,
    Section,
    TokenEstimator,
    chunk,
    chunk_report,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def cfg(**kw):
    c = ChunkingConfig.recommended()
    c.chunk_size_tokens = 256
    c.min_chunk_size_tokens = 64
    c.no_chunking_below_tokens = 0
    c.words_to_tokens_factor = 1.0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def one_section(text, **kw):
    return Document(sections=[Section(text=text, **kw)])


# --- chunk_report ------------------------------------------------------------

def test_report_fields():
    c = cfg(chunk_overlap_fraction=0.0)
    r = chunk_report(one_section(words(400)), c)
    assert r["chunk_count"] == 2
    assert r["min"] == 144
    assert r["max"] == 256
    assert r["total_tokens"] == 400
    assert r["boundary_crossings"] == 0
    assert r["sizes"] == [256, 144]


def test_report_empty_document():
    r = chunk_report(Document(sections=[]), cfg())
    assert r["chunk_count"] == 0
    assert r["min"] == 0
    assert r["max"] == 0
    assert r["avg"] == 0.0


def test_report_reflects_config_changes():
    # the adjust-inspect loop: same document, different size -> different
    # report, before anything is committed
    doc = one_section(words(400))
    small = chunk_report(doc, cfg(chunk_overlap_fraction=0.0))
    large = chunk_report(doc, cfg(chunk_overlap_fraction=0.0,
                                chunk_size_tokens=512))
    assert small["chunk_count"] == 2
    assert large["chunk_count"] == 1


def test_report_counts_boundary_crossings():
    # a well-behaved config never crosses: overlap stays inside sections
    c = cfg()
    doc = Document(sections=[
        Section(text=words(300, "a"), page=1),
        Section(text=words(300, "b"), page=2),
    ])
    assert chunk_report(doc, c)["boundary_crossings"] == 0


# --- collection / document overrides ------------------------------------------

def test_collection_overrides_apply_to_a_copy():
    c = cfg()
    c.collection_overrides = {"kb_app1": {"chunk_size_tokens": 512}}
    eff = c.for_collection("kb_app1")
    assert eff.chunk_size_tokens == 512
    assert eff.chunk_overlap_fraction == c.chunk_overlap_fraction  # rest kept
    assert c.chunk_size_tokens == 256  # base untouched


def test_collection_overrides_unknown_collection_gives_base():
    c = cfg()
    c.collection_overrides = {"kb_app1": {"chunk_size_tokens": 512}}
    assert c.for_collection("other").chunk_size_tokens == 256


def test_document_overrides_apply_to_a_copy():
    c = cfg()
    c.document_overrides = {"dense_spec": {"chunk_size_tokens": 256,
                                          "chunk_overlap_fraction": 0.0}}
    eff = c.for_document("dense_spec")
    assert eff.chunk_size_tokens == 256
    assert eff.chunk_overlap_fraction == 0.0


def test_overrides_that_violate_caps_raise():
    c = cfg()
    c.collection_overrides = {"bad": {"chunk_overlap_fraction": 0.5}}
    try:
        c.for_collection("bad")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- pluggable estimator ------------------------------------------------------

class DoubleEstimator(TokenEstimator):
    """A stand-in for a real tokenizer: 2 'tokens' per word."""

    def estimate(self, text: str) -> int:
        return 2 * len(text.split())


def test_pluggable_estimator_drives_sizing():
    c = cfg(chunk_overlap_fraction=0.0)
    # 400 words -> 800 est tokens; window = 128 words (256 est tokens)
    chunks = chunk(one_section(words(400)), c,
                   estimator=DoubleEstimator())
    # [128, 128, 16] word windows; 16-word tail (32 est < 64 min) merged
    assert [ch.tokens for ch in chunks] == [256, 256, 288]


def test_tokenizer_method_requires_explicit_estimator():
    c = cfg(token_method="tokenizer")
    try:
        chunk(one_section(words(400)), c)
        assert False, "expected ValueError"
    except ValueError:
        pass
