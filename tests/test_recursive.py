"""Cycle 3: recursive (default) strategy + structure handling (public API).

Golden tests use words_to_tokens_factor=1.0 (estimate == word count) and
size 256 (the range floor) so boundaries are exact literals.
"""
from chunking_system import (
    ChunkingConfig,
    Document,
    Section,
    chunk,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def recursive_config(**kw):
    c = ChunkingConfig.recommended()  # split_strategy is already recursive
    c.chunk_size_tokens = 256
    c.min_chunk_size_tokens = 64
    c.no_chunking_below_tokens = 0
    c.words_to_tokens_factor = 1.0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def one_section(text, **kw):
    return Document(sections=[Section(text=text, **kw)])


# --- separator preference (the recursive property) ----------------------------

def test_recursive_keeps_paragraphs_together():
    # 3 paragraphs of 100 words; 300 > 256 so a split is forced, but the
    # cut lands on paragraph boundaries, never mid-paragraph.
    p1, p2, p3 = words(100, "a"), words(100, "b"), words(100, "c")
    text = p1 + "\n\n" + p2 + "\n\n" + p3
    chunks = chunk(one_section(text),
                   recursive_config(chunk_overlap_fraction=0.0))
    assert len(chunks) == 2
    assert chunks[0].text == p1 + " " + p2  # P1+P2 fit in one 256 window
    assert chunks[1].text == p3


def test_recursive_contrast_fixed_mid_paragraph_cut():
    p1, p2, p3 = words(100, "a"), words(100, "b"), words(100, "c")
    text = p1 + "\n\n" + p2 + "\n\n" + p3
    c_fixed = recursive_config()
    c_fixed.split_strategy = "fixed"
    c_fixed.chunk_overlap_fraction = 0.0
    fixed = chunk(one_section(text), c_fixed)
    # fixed: [256, 44] -> 44 < 64 -> tail merged -> one 300-word chunk,
    # cutting through P3
    assert len(fixed) == 1
    assert fixed[0].text == " ".join(text.split())
    recursive = chunk(one_section(text), recursive_config())
    assert len(recursive) == 2  # clean paragraph-aligned chunks


def test_recursive_fallback_when_no_separator_fits():
    # 400 words, no separators: cuts only at the end
    chunks = chunk(one_section(words(400)),
                   recursive_config(chunk_overlap_fraction=0.0))
    assert [ch.tokens for ch in chunks] == [256, 144]


def test_recursive_overlap_golden():
    p1, p2, p3 = words(100, "a"), words(100, "b"), words(100, "c")
    text = p1 + "\n\n" + p2 + "\n\n" + p3
    all_words = text.split()
    c = recursive_config()  # overlap 0.25 -> 64 words
    chunks = chunk(one_section(text), c)
    assert chunks[0].text == " ".join(all_words[0:200])
    # next window starts 64 words back: [136:300]
    assert chunks[1].text == " ".join(all_words[136:300])


def test_recursive_is_the_default_strategy():
    c = ChunkingConfig.recommended()
    c.no_chunking_below_tokens = 0
    c.words_to_tokens_factor = 1.0
    assert c.split_strategy == "recursive"
    chunks = chunk(one_section(words(700)), c)  # 700 > 512 cap: splits
    assert len(chunks) == 2


# --- respect_structure -------------------------------------------------------

def test_respect_structure_false_chunks_whole_document():
    c = recursive_config(respect_structure=False,
                        chunk_overlap_fraction=0.0)
    doc = Document(sections=[
        Section(text=words(200, "a"), page=1),
        Section(text=words(200, "b"), page=2),
    ])
    chunks = chunk(doc, c)
    # one text of 400 words with a paragraph break at 200: the recursive
    # cut lands on the break -> [200, 200]
    assert [ch.tokens for ch in chunks] == [200, 200]
    assert chunks[0].metadata["page"] is None
    assert chunks[0].metadata["section_index"] == 0


# --- property tests -----------------------------------------------------------

def test_property_chunks_within_guard_and_boundaries():
    c = recursive_config()
    doc = one_section(words(1000))
    chunks = chunk(doc, c)
    for i, ch in enumerate(chunks):
        assert ch.tokens <= c.max_chunk_size_tokens
        if i < len(chunks) - 1:  # only a merged tail may exceed the target
            assert ch.tokens <= c.chunk_size_tokens


def test_property_no_words_cross_section_boundaries():
    c = recursive_config()
    doc = Document(sections=[
        Section(text=words(300, "a"), page=1),
        Section(text=words(300, "b"), page=2),
    ])
    chunks = chunk(doc, c)
    for ch in chunks:
        ws = set(ch.text.split())
        if ch.metadata["page"] == 1:
            assert not any(w.startswith("b") for w in ws)
        else:
            assert not any(w.startswith("a") for w in ws)


def test_property_recursive_deterministic():
    c = recursive_config()
    p1, p2 = words(150, "x"), words(150, "y")
    doc = one_section(p1 + "\n\n" + p2)
    assert [ch.text for ch in chunk(doc, c)] == [ch.text for ch in chunk(doc, c)]
