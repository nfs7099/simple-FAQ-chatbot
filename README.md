✨ Simple FAQ Chatbot (RAG)

Local or hosted LLMs • PDF ingestion • Modern UI • Theming • Sources & reranking

A local-first FAQ chatbot that lets you upload PDFs, builds a vector index, and answers questions with citations. Run fully offline via Ollama, or switch to OpenAI/Anthropic with API keys. Skinnable UI, clean API, and practical defaults.


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
