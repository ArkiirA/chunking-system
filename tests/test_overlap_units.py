"""Cycle 5: overlap unit correctness (tokens vs words) at the default
1.37 factor — the tanyamesin adoption exposed this.
"""
from chunking_system import ChunkingConfig, Document, Section, chunk


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def one_section(text, **kw):
    return Document(sections=[Section(text=text, **kw)])


def _shared_suffix_prefix(aw: list, bw: list) -> int:
    """Largest L with b[0:L] == a[-L:] (the overlap region)."""
    for L in range(min(len(aw), len(bw)), 0, -1):
        if aw[len(aw) - L:] == bw[:L]:
            return L
    return 0


def test_default_config_overlap_is_25_percent_in_token_units():
    # 2000 words at the recommended defaults (512 tokens, 1.37 factor,
    # 25% = 128-token overlap = 93 words):
    # windows [0:373], [280:653], [560:933], [840:1213], [1120:1493],
    # [1400:1773], [1680:2000] -> 7 chunks (worked example, factor 1.37)
    c = ChunkingConfig.recommended()
    c.no_chunking_below_tokens = 0
    chunks = chunk(one_section(words(2000)), c)
    assert len(chunks) == 7


def test_default_config_overlap_word_bound():
    # the shared tail between consecutive chunks must be at most
    # int(128 / 1.37) = 93 words (128 tokens = 25% of 512)
    c = ChunkingConfig.recommended()
    c.no_chunking_below_tokens = 0
    chunks = chunk(one_section(words(2000)), c)
    for a, b in zip(chunks, chunks[1:]):
        shared = _shared_suffix_prefix(a.text.split(), b.text.split())
        assert shared <= 93, (shared, a.tokens, b.tokens)


def test_pluggable_estimator_overlap_is_in_its_own_units():
    from chunking_system import TokenEstimator

    class DoubleEstimator(TokenEstimator):
        """2 'tokens' per word: the 128-token overlap is 64 words."""
        def estimate(self, text: str) -> int:
            return 2 * len(text.split())

    c = ChunkingConfig.recommended()
    c.no_chunking_below_tokens = 0
    chunks = chunk(one_section(words(2000)), c,
                   estimator=DoubleEstimator())
    # k = 256 words, overlap 64 words:
    # [0:256], [192:448], ..., [1728:1984], [1920:2000] -> 11 chunks
    assert len(chunks) == 11
    for a, b in zip(chunks, chunks[1:]):
        shared = _shared_suffix_prefix(a.text.split(), b.text.split())
        assert shared <= 64, shared
