"""Cycle 1: core types, config surface, word estimate (public API only)."""
import pytest

from chunking_system import (
    Chunk,
    ChunkingConfig,
    Document,
    Section,
    WordEstimator,
)


def make_section(text, page=None, section_title=None, doc_title=None):
    return Section(text=text, page=page, section_title=section_title,
                  doc_title=doc_title)


# --- core types -----------------------------------------------------------

def test_section_carries_optional_structure():
    s = make_section("hello world", page=3, section_title="Setup",
                    doc_title="Manual")
    assert s.text == "hello world"
    assert s.page == 3
    assert s.section_title == "Setup"
    assert s.doc_title == "Manual"


def test_section_structure_is_optional():
    s = make_section("hello world")
    assert s.page is None
    assert s.section_title is None
    assert s.doc_title is None


def test_document_is_an_ordered_list_of_sections():
    doc = Document(sections=[make_section("a"), make_section("b")])
    assert [s.text for s in doc.sections] == ["a", "b"]


def test_word_estimate_is_word_count_times_factor():
    est = WordEstimator(factor=1.37)
    # 5 words * 1.37 = 6.85 -> 7 (ceil: an estimate must never undercount)
    assert est.estimate("one two three four five") == 7
    assert est.estimate("") == 0


# --- config surface -------------------------------------------------------

def test_recommended_factory_gives_research_defaults():
    c = ChunkingConfig.recommended()
    assert c.chunk_size_tokens == 512
    assert c.chunk_overlap_fraction == 0.25
    assert c.split_strategy == "recursive"
    assert c.separators == ["\n\n", "\n", " ", ""]
    assert c.respect_structure is True
    assert c.min_chunk_size_tokens == 128
    assert c.max_chunk_size_tokens == 2048
    assert c.embedding_model_context == 8192
    assert c.token_method == "word_estimate"
    assert c.words_to_tokens_factor == 1.37
    assert c.no_chunking_below_tokens == 512
    assert c.include_title is True
    assert c.include_section is True


def test_recommended_config_passes_its_own_validation():
    ChunkingConfig.recommended().validate()  # must not raise


def test_variable_spec_exposes_recommended_and_range():
    c = ChunkingConfig.recommended()
    spec = c.variable_spec("chunk_size_tokens")
    assert spec["recommended"] == 512
    assert spec["min"] == 256
    assert spec["max"] == 1024
    spec = c.variable_spec("chunk_overlap_fraction")
    assert spec["recommended"] == 0.25
    assert spec["max"] == 0.25  # hard cap is the range max


def test_validate_rejects_overlap_above_hard_cap():
    c = ChunkingConfig.recommended()
    c.chunk_overlap_fraction = 0.5
    with pytest.raises(ValueError, match="overlap"):
        c.validate()


def test_validate_rejects_max_chunk_above_90_percent_of_model_context():
    c = ChunkingConfig.recommended()
    c.max_chunk_size_tokens = int(0.91 * c.embedding_model_context)
    with pytest.raises(ValueError, match="context"):
        c.validate()


def test_validate_rejects_min_above_chunk_size():
    c = ChunkingConfig.recommended()
    c.min_chunk_size_tokens = 600
    with pytest.raises(ValueError, match="min"):
        c.validate()


def test_validate_rejects_chunk_size_below_range():
    c = ChunkingConfig.recommended()
    c.chunk_size_tokens = 128
    with pytest.raises(ValueError, match="chunk_size"):
        c.validate()


def test_validate_rejects_factor_out_of_range():
    c = ChunkingConfig.recommended()
    c.words_to_tokens_factor = 2.0
    with pytest.raises(ValueError, match="factor"):
        c.validate()


def test_unimplemented_strategies_raise():
    from chunking_system import chunk
    c = ChunkingConfig.recommended()
    c.split_strategy = "semantic"
    doc = Document(sections=[make_section("x " * 500)])
    with pytest.raises(NotImplementedError):
        chunk(doc, c)
    c.split_strategy = "hierarchical"
    with pytest.raises(NotImplementedError):
        chunk(doc, c)


def test_unknown_strategy_rejected_by_validate():
    c = ChunkingConfig.recommended()
    c.split_strategy = "mystery"
    with pytest.raises(ValueError, match="strategy"):
        c.validate()
