"""Integration tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client. Skips if OPENAI_API_KEY is not set."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")
    from src.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "chunks_indexed" in data
        assert data["status"] == "online"
        assert data["service"] == "ops-copilot"
        assert isinstance(data["chunks_indexed"], int)


class TestAskEndpoint:
    def test_ask_returns_200(self, client):
        response = client.post("/ask", json={"question": "What is the motion range of the J1 axis?"})
        assert response.status_code == 200

    def test_ask_response_schema(self, client):
        response = client.post("/ask", json={"question": "How to recover from an E-stop?"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_ask_returns_sources(self, client):
        response = client.post("/ask", json={"question": "What is the payload capacity?"})
        data = response.json()
        if len(data["sources"]) > 0:
            source = data["sources"][0]
            assert "text" in source
            assert "source" in source
            assert "score" in source

    def test_ask_with_history(self, client):
        response = client.post("/ask", json={
            "question": "What about the J2 axis?",
            "history": [
                {"question": "What is J1 range?", "answer": "±170 degrees"}
            ]
        })
        assert response.status_code == 200

    def test_ask_empty_question(self, client):
        """Empty string should still return a response (not crash)."""
        response = client.post("/ask", json={"question": ""})
        # FastAPI may return 422 for validation or 200 with empty answer — either is acceptable
        assert response.status_code in [200, 422]
