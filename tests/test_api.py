import unittest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        
        # Check that the response has expected structure
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("pdf_count", data)
        self.assertIn("vector_db_initialized", data)
        self.assertIn("llm_initialized", data)
        self.assertIn("pdfs", data)

        # Check types
        self.assertIsInstance(data["status"], str)
        self.assertIsInstance(data["pdf_count"], int)
        self.assertIsInstance(data["vector_db_initialized"], bool)
        self.assertIsInstance(data["llm_initialized"], bool)
        self.assertIsInstance(data["pdfs"], list)

    def test_querty_endpoint_valid(self):
        # This test is expected top fail if the system is not ready
        response = self.client.post("/api/query", json={"query": "What is e-waste?"})

        if response.status_code == 503:
            self.assertIn("RAG pipeline not initialized", response.json()["detail"])
            return
        
        # If the system is ok we expect a 200 OK with the correct structure
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("answer", data)
        self.assertIn("sources", data)

        self.assertIsInstance(data["answer"], str)
        self.assertIsInstance(data["sources"], list)

        # If there are sources check their structure
        if data["sources"]:
            source = data["sources"][0]
            self.assertIn("source", source)
            self.assertIn("content", source)
            self.assertIn("page", source)

if __name__ == "__main__":
    unittest.main()