"""Unit tests for the VectorStore retriever."""
import pytest
from unittest.mock import patch, MagicMock
from src.rag.retriever import VectorStore


class TestChunkText:
    """Tests for the chunk_text method."""

    def test_basic_chunking(self):
        """Chunks split at sentence boundaries."""
        store = VectorStore.__new__(VectorStore)  # Skip __init__
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

    def test_chunk_character_truncation(self):
        """No chunk exceeds 4000 characters."""
        store = VectorStore.__new__(VectorStore)
        long_sentence = "A" * 5000
        text = long_sentence + ". " + long_sentence
        chunks = store.chunk_text(text, max_words=10000)
        for chunk in chunks:
            assert len(chunk) <= 4000


class TestVectorStoreSearch:
    """Mocked unit tests for the RAG search pipeline."""

    @pytest.fixture
    def mock_store(self):
        """Create a VectorStore with mocked OpenAI and ChromaDB clients."""
        with patch('src.rag.retriever.OpenAI') as MockOpenAI, \
             patch('src.rag.retriever.chromadb.PersistentClient') as MockChroma:
            
            # Setup Chroma Mock
            mock_collection = MagicMock()
            MockChroma.return_value.get_or_create_collection.return_value = mock_collection
            
            store = VectorStore()
            store.collection = mock_collection
            store.client = MockOpenAI.return_value
            return store

    def test_get_embedding_success(self, mock_store):
        """Test embedding generation isolates correctly."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_store.client.embeddings.create.return_value = mock_response

        emb = mock_store._get_embedding("test text")
        
        assert emb == [0.1, 0.2, 0.3]
        mock_store.client.embeddings.create.assert_called_once_with(
            input="test text",
            model="text-embedding-3-small"
        )

    @patch('src.rag.retriever.BM25Okapi')
    def test_hybrid_search(self, mock_bm25, mock_store):
        """Test that hybrid search combines Vector and BM25 results correctly."""
        # Setup BM25 Mock
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [1.5, 0.0, 2.5] # Docs 0, 1, 2
        mock_bm25.return_value = mock_bm25_instance
        
        # Setup mock chunks
        mock_store.chunks = ["Doc A about robots", "Doc B unrelated", "Doc C about robots"]
        mock_store.chunk_sources = [{"source": "manual.pdf"}, {"source": "manual.pdf"}, {"source": "manual.pdf"}]
        mock_store.bm25 = mock_bm25_instance

        # Setup Vector Mock
        mock_store.collection.query.return_value = {
            "ids": [["doc_0", "doc_2"]],
            "documents": [["Doc A about robots", "Doc C about robots"]],
            "metadatas": [[{"source": "manual.pdf"}, {"source": "manual.pdf"}]]
        }

        # Mock embedding
        with patch.object(mock_store, '_get_embedding', return_value=[0.1]*1536):
            results = mock_store.search("robot", top_k=2)

        assert len(results) == 2
        assert "text" in results[0]
        assert "score" in results[0]
        # Should be fused and sorted by score
        assert results[0]["score"] >= results[1]["score"]

    def test_empty_search_query(self, mock_store):
        """Empty query should gracefully return empty results."""
        results = mock_store.search("")
        assert results == []
        mock_store.collection.query.assert_not_called()
