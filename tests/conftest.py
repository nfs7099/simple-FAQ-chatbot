# tests/conftest.py
import os
import sys
import types
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Tiny stubs (no downloads) ----
class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[(len(t) % 7) / 10.0] * 8 for t in texts]
    def embed_query(self, text):
        return [(len(text) % 7) / 10.0] * 8

class FakeLLM:
    def __init__(self, content="(test) Answer from provided files."):
        self._content = content
    def invoke(self, prompt):
        return types.SimpleNamespace(content=self._content)

@pytest.fixture(autouse=True)
def test_env(monkeypatch, tmp_path):
    """Per-test env. Do NOT import app/modules before this runs."""
    pdf_dir = tmp_path / "raw"
    vectordb_dir = tmp_path / "vector_db"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PDF_DIR", str(pdf_dir))
    monkeypatch.setenv("VECTOR_DB_PATH", str(vectordb_dir))
    monkeypatch.setenv("ALLOW_DANGEROUS_DESERIALIZATION", "true")
    monkeypatch.setenv("USE_TOKEN_SPLITTER", "false")   # faster tests (no transformers)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")        # irrelevant; we stub the LLM
    monkeypatch.setenv("ALLOWED_EXTENSIONS", ".pdf")
    monkeypatch.setenv("DEBUG", "false")

    # If you added auth/rate-limits, keep them off in tests:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("REQUIRE_API_KEY_FOR_ADMIN", "false")
    monkeypatch.setenv("REQUIRE_API_KEY_FOR_QUERY", "false")

    yield  # modules are imported in the client fixture

@pytest.fixture()
def client(monkeypatch, test_env):
    """
    Reload modules AFTER env is set so each test sees its own temp dirs.
    Then re-apply stubs to the freshly loaded classes.
    """
    # Reload in dependency order
    import configs.config as cfg
    importlib.reload(cfg)

    import src.vector_db.vector_store as vs_mod
    importlib.reload(vs_mod)

    import src.rag_pipeline.rag_chain as rc_mod
    importlib.reload(rc_mod)

    import src.api.app as app_mod
    importlib.reload(app_mod)

    # Patch heavy bits on the reloaded classes
    def _fake_init_embedding_model(self):
        self.embedding_model = DummyEmbeddings()
    monkeypatch.setattr(vs_mod.VectorStore, "_init_embedding_model", _fake_init_embedding_model, raising=False)

    def _fake_build_llm(self, provider: str, temperature: float):
        return FakeLLM()
    monkeypatch.setattr(rc_mod.RAGPipeline, "_build_llm", _fake_build_llm, raising=False)

    # Reset singleton so each test starts clean
    app_mod.rag_pipeline = None

    from pypdf import PdfWriter  # ensure available for make_pdf

    with TestClient(app_mod.app) as c:
        yield c

@pytest.fixture()
def make_pdf(tmp_path):
    """Create a 1-page, valid PDF under the current test's PDF_DIR."""
    from pypdf import PdfWriter
    def _make(name: str = "sample.pdf"):
        pdf_dir = Path(os.environ["PDF_DIR"])
        path = pdf_dir / name
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(path, "wb") as f:
            writer.write(f)
        return path
    return _make

@pytest.fixture()
def make_pdf(tmp_path):
    """Create a 1-page PDF with some text under the current test's PDF_DIR."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    def _make(name: str = "sample.pdf", text: str = "Hello from tests"):
        pdf_dir = Path(os.environ["PDF_DIR"])
        path = pdf_dir / name
        c = canvas.Canvas(str(path), pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, text)  # 1 inch margin, near top
        c.showPage()
        c.save()
        return path
    return _make