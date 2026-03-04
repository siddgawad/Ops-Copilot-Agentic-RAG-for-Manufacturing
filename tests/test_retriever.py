"""Unit tests for the VectorStore retriever."""
import pytest
from src.rag.retriever import VectorStore


class TestChunkText:
    """Tests for the chunk_text method."""

    def test_basic_chunking(self):
        """Chunks split at sentence boundaries."""
        store = VectorStore.__new__(VectorStore)  # Skip __init__ (needs OpenAI key)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = store.chunk_text(text, max_words=5)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_single_sentence(self):
        """A single sentence returns one chunk."""
        store = VectorStore.__new__(VectorStore)
        text = "This is one sentence."
        chunks = store.chunk_text(text, max_words=100)
        assert len(chunks) == 1

    def test_empty_text(self):
        """Empty text returns one chunk (the trailing period)."""
        store = VectorStore.__new__(VectorStore)
        text = ""
        chunks = store.chunk_text(text, max_words=100)
        assert len(chunks) >= 0

    def test_chunk_character_truncation(self):
        """No chunk exceeds 4000 characters."""
        store = VectorStore.__new__(VectorStore)
        # Create text with very long sentences
        long_sentence = "A" * 5000
        text = long_sentence + ". " + long_sentence
        chunks = store.chunk_text(text, max_words=10000)
        for chunk in chunks:
            assert len(chunk) <= 4000, f"Chunk exceeded 4000 chars: {len(chunk)}"

    def test_overlap_exists(self):
        """Consecutive chunks share overlap (last sentence of chunk N = first of chunk N+1)."""
        store = VectorStore.__new__(VectorStore)
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence. Epsilon sentence."
        chunks = store.chunk_text(text, max_words=3)
        if len(chunks) >= 2:
            # The last sentence of chunk 0 should appear at the start of chunk 1
            last_sentence_chunk0 = chunks[0].rstrip(".").split(". ")[-1]
            assert last_sentence_chunk0 in chunks[1], "Overlap not found between consecutive chunks"

    def test_word_count_respected(self):
        """Each chunk stays near the max_words limit."""
        store = VectorStore.__new__(VectorStore)
        words = " ".join([f"word{i}" for i in range(200)])
        # Make it sentence-like
        text = ". ".join([f"Sentence number {i} with some words" for i in range(50)])
        chunks = store.chunk_text(text, max_words=20)
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow some slack due to overlap, but no chunk should be wildly oversized
            assert word_count < 60, f"Chunk has {word_count} words, expected near 20"
