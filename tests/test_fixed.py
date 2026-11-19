"""Cycle 2: fixed strategy end-to-end (public API only).

Golden tests use words_to_tokens_factor=1.0 so the estimate equals the word
count and expected boundaries are exact literals. All sizes respect the
valid config ranges (size 256-1024, min 64-256) and the settled
undersized-tail merge decision.
"""
from chunking_system import (
    ChunkingConfig,
    Document,
    Section,
    chunk,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def fixed_config(**kw):
    c = ChunkingConfig.recommended()
    c.split_strategy = "fixed"
    c.chunk_size_tokens = 256
    c.min_chunk_size_tokens = 64
    c.no_chunking_below_tokens = 0  # chunking tests: no whole-section floor
    c.words_to_tokens_factor = 1.0  # estimate == word count
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def one_section(text, **kw):
    return Document(sections=[Section(text=text, **kw)])


# --- basic splitting ---------------------------------------------------------

def test_fixed_short_text_is_one_chunk():
    chunks = chunk(one_section(words(50)), fixed_config())
    assert len(chunks) == 1
    assert chunks[0].text == words(50)
    assert chunks[0].tokens == 50  # factor 1.0


def test_fixed_long_text_golden_boundaries_no_overlap():
    c = fixed_config(chunk_overlap_fraction=0.0)
    all_words = words(400).split()
    chunks = chunk(one_section(words(400)), c)
    assert [ch.tokens for ch in chunks] == [256, 144]
    assert chunks[0].text == " ".join(all_words[0:256])
    assert chunks[1].text == " ".join(all_words[256:400])


def test_fixed_overlap_golden_boundaries():
    c = fixed_config(chunk_overlap_fraction=0.25)  # 64 words of overlap
    all_words = words(400).split()
    chunks = chunk(one_section(words(400)), c)
    # windows [0:256], [192:400]
    assert chunks[0].text == " ".join(all_words[0:256])
    assert chunks[1].text == " ".join(all_words[192:400])


def test_fixed_zero_overlap_no_repeated_words():
    c = fixed_config(chunk_overlap_fraction=0.0)
    chunks = chunk(one_section(words(400)), c)
    seen = []
    for ch in chunks:
        seen += ch.text.split()
    assert len(seen) == len(set(seen)) == 400


# --- no_chunking_below -------------------------------------------------------

def test_no_chunking_below_keeps_short_section_whole():
    c = fixed_config(no_chunking_below_tokens=512)
    chunks = chunk(one_section(words(100)), c)
    assert len(chunks) == 1
    assert chunks[0].text == words(100)


def test_no_chunking_below_zero_always_chunks():
    c = fixed_config(no_chunking_below_tokens=0)
    chunks = chunk(one_section(words(100)), c)
    assert len(chunks) == 1
    assert chunks[0].text == words(100)  # under the cap: one chunk


def test_no_chunking_below_just_over_floor_splits():
    c = fixed_config(no_chunking_below_tokens=50,
                    chunk_overlap_fraction=0.0)
    chunks = chunk(one_section(words(320)), c)
    assert [ch.tokens for ch in chunks] == [256, 64]


# --- undersized tail: merge (settled decision) --------------------------------

def test_undersized_tail_is_merged_into_previous():
    c = fixed_config(chunk_overlap_fraction=0.0)
    chunks = chunk(one_section(words(286)), c)
    # 286 -> [256, 30]; 30 < 64 -> merged back: one 286-word chunk
    assert len(chunks) == 1
    assert chunks[0].text == words(286)


def test_tail_merge_respects_max_guard():
    c = fixed_config(chunk_overlap_fraction=0.0, max_chunk_size_tokens=280)
    chunks = chunk(one_section(words(286)), c)
    # merge would give 286 > 280 guard -> tail stays separate
    assert [ch.tokens for ch in chunks] == [256, 30]


# --- structure, metadata, embedded text ---------------------------------------

def test_metadata_carries_structure_and_index():
    c = fixed_config()
    doc = Document(sections=[
        Section(text=words(50), page=7, section_title="Setup",
               doc_title="HyCap 160"),
        Section(text=words(60), page=8, section_title="Operation",
               doc_title="HyCap 160"),
    ])
    chunks = chunk(doc, c)
    assert len(chunks) == 2
    m0, m1 = chunks[0].metadata, chunks[1].metadata
    assert m0["page"] == 7 and m1["page"] == 8
    assert m0["section"] == "Setup" and m1["section"] == "Operation"
    assert m0["title"] == "HyCap 160"
    assert m0["chunk_index"] == 0 and m1["chunk_index"] == 0
    assert m0["section_index"] == 0 and m1["section_index"] == 1


def test_chunk_index_counts_within_section():
    c = fixed_config(chunk_overlap_fraction=0.0)
    doc = one_section(words(400))
    chunks = chunk(doc, c)
    assert [ch.metadata["chunk_index"] for ch in chunks] == [0, 1]


def test_overlap_never_crosses_section_boundaries():
    c = fixed_config()
    doc = Document(sections=[
        Section(text=words(256, "a"), page=1),
        Section(text=words(256, "b"), page=2),
    ])
    chunks = chunk(doc, c)
    # each section is exactly at the cap: one chunk each, no cross-leak
    assert len(chunks) == 2
    assert chunks[1].text.startswith("a0") is False
    assert chunks[1].text.startswith("b0")


def test_embedded_text_prepends_title_and_section():
    c = fixed_config()
    doc = one_section(words(50), page=1, section_title="Setup",
                     doc_title="HyCap 160")
    ch = chunk(doc, c)[0]
    assert ch.embedded_text == "HyCap 160\nSetup\n" + words(50)


def test_embedded_text_flags_off_gives_raw_text():
    c = fixed_config(include_title=False, include_section=False)
    doc = one_section(words(50), section_title="Setup", doc_title="T")
    ch = chunk(doc, c)[0]
    assert ch.embedded_text == words(50)


# --- determinism -------------------------------------------------------------

def test_chunking_is_deterministic():
    c = fixed_config()
    doc = one_section(words(333))
    first = chunk(doc, c)
    second = chunk(doc, c)
    assert [ch.text for ch in first] == [ch.text for ch in second]
    assert [ch.metadata for ch in first] == [ch.metadata for ch in second]
