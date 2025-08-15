# tests/test_api.py
from pathlib import Path

def test_status_before_indexing(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    # No index yet
    assert body["vector_db_initialized"] in (False, 0)


def test_reindex_full_then_query(client, make_pdf):
    # Create one test PDF
    make_pdf("doc1.pdf")

    # Full rebuild
    r = client.post("/api/reindex", params={"mode": "full"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Status should show initialized and 1 PDF
    s = client.get("/api/status").json()
    assert s["vector_db_initialized"] in (True, 1)
    assert s["pdf_count"] == 1

    # Query should return an answer and (usually) at least one source
    q = client.post("/api/query", json={"query": "What is e-waste?"})
    assert q.status_code == 200
    body = q.json()
    assert isinstance(body["answer"], str) and len(body["answer"]) > 0
    # Sources may be 0 if blank page; presence of key is what we check
    assert "sources" in body


def test_incremental_reindex_adds_new_only(client, make_pdf):
    # Start with one PDF and full rebuild
    make_pdf("doc1.pdf")
    client.post("/api/reindex", params={"mode": "full"})

    # Add a second PDF and run incremental
    make_pdf("doc2.pdf")
    r = client.post("/api/reindex")  # default: incremental scan-and-sync
    assert r.status_code == 200
    result = r.json()["result"]
    # Should be incremental mode (not forced full rebuild)
    assert result["mode"] != "rebuild-failed"
    # After incremental, status should show 2 PDFs tracked
    s = client.get("/api/status").json()
    assert s["pdf_count"] == 2


def test_upload_rejects_non_pdf(client):
    # Pretend upload of a .txt file (wrong extension)
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
    assert "Only" in r.json()["detail"] or "PDF" in r.json()["detail"]

    # Also reject a .pdf name with non-PDF content
    files = {"file": ("fake.pdf", b"NOT_A_PDF", "application/pdf")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
    assert "not a valid PDF" in r.json()["detail"]
