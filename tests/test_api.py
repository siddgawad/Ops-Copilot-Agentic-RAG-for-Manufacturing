"""Integration tests for the FastAPI endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# We must patch before importing the app because it instantiates VectorStore globally.
with patch("src.rag.retriever.VectorStore.load_documents_from_folder"), \
     patch("src.rag.retriever.VectorStore.__init__", return_value=None):
    from src.main import app, db


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        db.raw_chunks = ["chunk1", "chunk2"]
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        db.raw_chunks = ["chunk1", "chunk2", "chunk3"]
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "online"
        assert data["service"] == "ops-copilot"
        assert data["chunks_indexed"] == 3


class TestAskEndpoint:
    @patch('src.main.generate_answer')
    def test_ask_returns_mocked_results(self, mock_generate, client):
        """Test the full /ask endpoint with mocked RAG components."""
        
        # Setup the mocks
        db.search = MagicMock(return_value=[
            {"text": "mocked chunk about payload", "source": "manual.pdf", "score": 0.95}
        ])
        mock_generate.return_value = "The payload is 20kg."

        # Make the request
        response = client.post("/ask", json={"question": "What is the payload?"})

        assert response.status_code == 200
        
        data = response.json()
        assert data["answer"] == "The payload is 20kg."
        assert len(data["sources"]) == 1
        
        source = data["sources"][0]
        assert "mocked chunk" in source["text"]
        assert source["source"] == "manual.pdf"
        assert source["score"] == 0.95

        # Verify our mocks were called correctly by the endpoint
        db.search.assert_called_once_with(query="What is the payload?", n_results=3)
        mock_generate.assert_called_once_with(
            "What is the payload?", 
            ["mocked chunk about payload"], 
            history=[]
        )

    @patch('src.main.generate_answer')
    def test_ask_with_history(self, mock_generate, client):
        """Test the endpoint handles conversation history correctly."""
        
        db.search = MagicMock(return_value=[])
        mock_generate.return_value = "It covers 150 degrees."

        response = client.post("/ask", json={
            "question": "What about the J2 axis?",
            "history": [
                {"question": "What is J1 range?", "answer": "±170 degrees"}
            ]
        })
        
        assert response.status_code == 200
        
        # History should be passed to the generator
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert len(call_kwargs["history"]) == 1
        assert call_kwargs["history"][0]["question"] == "What is J1 range?"

    def test_ask_empty_question(self, client):
        """Empty query should be caught by FastAPI standard validation (422) or handled."""
        response = client.post("/ask", json={"question": ""})
        # Our endpoint might let it through to empty search, or FastAPI might block it
        assert response.status_code in [200, 422]
