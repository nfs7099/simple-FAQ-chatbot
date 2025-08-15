✨ Simple FAQ Chatbot (RAG)

Local or hosted LLMs • PDF ingestion • Modern UI • Theming • Sources & reranking

A local-first FAQ chatbot that lets you upload PDFs, builds a vector index, and answers questions with citations. Run fully offline via Ollama, or switch to OpenAI/Anthropic with API keys. Skinnable UI, clean API, and practical defaults.

Features

✅ Local or hosted LLMs: Ollama (local), or OpenAI / Anthropic (cloud).

✅ RAG pipeline: HuggingFace embeddings + FAISS + optional reranker.

✅ PDF ingestion: upload control; incremental and full rebuild indexing.

✅ Token-aware chunking: smarter splits with configurable overlap.

✅ Citations: expandable source snippets with page hints.

✅ Polished UI: React + FastAPI; themeable (emerald/indigo/slate/rose/amber/teal).

✅ Health/ready endpoints for container orchestration.

✅ Safer uploads: PDF-only (extension + magic bytes), size limits, path-traversal safe.

Architecture
                        ┌──────────────────────────────────────────────────┐
                        │                    Frontend                      │
                        │  React UI (static) + Theming via CSS variables   │
                        └───────▲───────────────────────────────▲──────────┘
                                │                               │
                     UI config  │           REST API            │ Chat & status
         /api/ui-config + theme │      /api/query /api/status   │ updates
                                │                               │
┌───────────────────────────────┴───────────────┐  ┌────────────┴─────────────────┐
│                 FastAPI backend               │  │           RAG Pipeline        │
│  - upload, index, query, status endpoints     │  │  Embeddings + FAISS + rerank  │
│  - PDF validation (ext + magic bytes + size)  │  │  Token-based chunker          │
└───────────────────────────────┬───────────────┘  └────────────┬─────────────────┘
                                │                               │
                                │                               │
                       ┌────────▼─────────┐             ┌───────▼────────────────┐
                       │ Vector store     │             │ Language models         │
                       │ FAISS on disk    │             │ Ollama / OpenAI / Anth.│
                       └──────────────────┘             └─────────────────────────┘
