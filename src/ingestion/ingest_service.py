import json
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from configs.config import (
    PDF_DIR, VECTOR_DB_PATH, MANIFEST_PATH,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    USE_TOKEN_SPLITTER, TOKENIZER_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    TOKEN_CHUNK_SIZE, TOKEN_CHUNK_OVERLAP,
    REBUILD_ON_MODIFICATION, REBUILD_ON_DELETE, ALLOWED_EXTENSIONS
)
from src.vector_db.vector_store import VectorStore

def _sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _is_allowed_pdf_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS

def _has_pdf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False

def scan_dir(pdf_dir: Path) -> Dict[str, Tuple[Path, str, int, float]]:
    """Return mapping filename -> (path, sha256, size, mtime) for valid PDFs only."""
    out: Dict[str, Tuple[Path, str, int, float]] = {}
    for p in pdf_dir.iterdir():
        if not _is_allowed_pdf_path(p):
            continue
        if not _has_pdf_magic(p):
            continue
        try:
            out[p.name] = (p, _sha256_file(p), p.stat().st_size, p.stat().st_mtime)
        except Exception:
            continue
    return out


@dataclass
class DocEntry:
    path: str
    sha256: str
    size: int
    mtime: float
    n_chunks: int
    added_at: str

@dataclass
class Manifest:
    embedder_provider: str
    embedder_model: str
    token_splitter: bool
    tokenizer_name: str
    char_chunk_size: int
    char_chunk_overlap: int
    token_chunk_size: int | None
    token_chunk_overlap: int | None
    docs: Dict[str, DocEntry]  

    @staticmethod
    def empty() -> "Manifest":
        model = EMBEDDING_MODEL if EMBEDDING_PROVIDER == "huggingface" else OPENAI_EMBEDDING_MODEL
        return Manifest(
            embedder_provider=EMBEDDING_PROVIDER,
            embedder_model=model,
            token_splitter=USE_TOKEN_SPLITTER,
            tokenizer_name=TOKENIZER_NAME,
            char_chunk_size=CHUNK_SIZE,
            char_chunk_overlap=CHUNK_OVERLAP,
            token_chunk_size=TOKEN_CHUNK_SIZE,
            token_chunk_overlap=TOKEN_CHUNK_OVERLAP,
            docs={}
        )

def load_manifest(path: Path = Path(MANIFEST_PATH)) -> Manifest:
    if not path.exists():
        return Manifest.empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["docs"] = {k: DocEntry(**v) for k, v in data.get("docs", {}).items()}
    return Manifest(**data)

def save_manifest(m: Manifest, path: Path = Path(MANIFEST_PATH)) -> None:
    serial = asdict(m)
    serial["docs"] = {k: asdict(v) for k, v in m.docs.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")

def config_has_changed(m: Manifest) -> bool:
    ref = Manifest.empty()
    return any([
        m.embedder_provider != ref.embedder_provider,
        m.embedder_model != ref.embedder_model,
        bool(m.token_splitter) != bool(ref.token_splitter),
        m.tokenizer_name != ref.tokenizer_name,
        int(m.char_chunk_size) != int(ref.char_chunk_size),
        int(m.char_chunk_overlap) != int(ref.char_chunk_overlap),
        (m.token_chunk_size or -1) != (ref.token_chunk_size or -1),
        (m.token_chunk_overlap or -1) != (ref.token_chunk_overlap or -1),
    ])

def scan_dir(pdf_dir: Path) -> Dict[str, Tuple[Path, str, int, float]]:
    """Return mapping filename -> (path, sha256, size, mtime)."""
    out: Dict[str, Tuple[Path, str, int, float]] = {}
    for p in pdf_dir.glob("*.pdf"):
        try:
            out[p.name] = (p, _sha256_file(p), p.stat().st_size, p.stat().st_mtime)
        except Exception:
            continue
    return out

def diff_manifest(m: Manifest, disk: Dict[str, Tuple[Path, str, int, float]]):
    existing = set(m.docs.keys())
    current = set(disk.keys())

    new_files = list(current - existing)
    deleted_files = list(existing - current)
    modified_files = [
        fn for fn in (existing & current)
        if m.docs[fn].sha256 != disk[fn][1] or int(m.docs[fn].size) != int(disk[fn][2])
    ]
    return new_files, modified_files, deleted_files

class IngestService:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def ingest_new_only(self, pdf_dir: Path = Path(PDF_DIR)) -> dict:
        """
        Incrementally ingest only brand-new files (no rebuild).
        Returns stats dict.
        """
        m = load_manifest()
        disk = scan_dir(pdf_dir)
        new_files, modified_files, deleted_files = diff_manifest(m, disk)

        added_chunks = 0
        added_files = []

        for fn in new_files:
            path, sha, size, mt = disk[fn]
            chunks = self.vs.load_and_split_pdf(path)
            if not chunks:
                continue
            ok = self.vs.add_documents(chunks)
            if ok:
                added_chunks += len(chunks)
                added_files.append(fn)
                m.docs[fn] = DocEntry(
                    path=str(path), sha256=sha, size=size, mtime=mt,
                    n_chunks=len(chunks), added_at=_now_iso()
                )

        save_manifest(m)
        return {
            "mode": "incremental-new",
            "added_files": added_files,
            "added_chunks": added_chunks,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
        }

    def scan_and_sync(self, pdf_dir: Path = Path(PDF_DIR)) -> dict:
        """
        Bring index in sync with folder:
        - If config changed OR modified/deleted detected (and policy says so) => full rebuild
        - Else add new files incrementally
        """
        m = load_manifest()
        disk = scan_dir(pdf_dir)
        new_files, modified_files, deleted_files = diff_manifest(m, disk)

        if config_has_changed(m) or (modified_files and REBUILD_ON_MODIFICATION) or (deleted_files and REBUILD_ON_DELETE):
            # rebuild from scratch for safety
            ok = self._full_rebuild(pdf_dir)
            status = "full-rebuild" if ok else "rebuild-failed"
            return {
                "mode": status,
                "added_files": list(disk.keys()),
                "modified_files": modified_files,
                "deleted_files": deleted_files,
            }

        # incremental add ofnew files
        return self.ingest_new_only(pdf_dir)

    def _full_rebuild(self, pdf_dir: Path) -> bool:
        # use the existing vector_store rebuild path
        ok = self.vs.load_documents(str(pdf_dir))
        if not ok:
            return False

        # recreating manifest again
        disk = scan_dir(pdf_dir)
        m = Manifest.empty()
        for fn, (p, sha, size, mt) in disk.items():
            m.docs[fn] = DocEntry(
                path=str(p), sha256=sha, size=size, mtime=mt, n_chunks=0, added_at=_now_iso()
            )
        save_manifest(m)
        return True
