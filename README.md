# ResearchForge — Agentic AI Research Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://researchforge-research-ai.streamlit.app/)

🔗 **[Live Demo](https://researchforge-research-ai.streamlit.app/)**  
💻 **[GitHub Repository](https://github.com/piyushbhati210/ResearchForge)**

An **Agentic AI research assistant** for exploring academic literature using **Retrieval-Augmented Generation (RAG), semantic search, SBERT embeddings, FAISS, LangChain, and LLMs**.

ResearchForge combines a conventional **Simple RAG** pipeline with a custom **ReAct-inspired multi-step retrieval workflow** for research-oriented question answering, research-gap identification, methodology extraction, and cross-paper comparison.

> **Project type:** Academic research prototype for MSc Data Science & Spatial Analytics.

---

## 🚀 Key Features

- 📄 Academic PDF ingestion and preprocessing
- 🔎 Semantic search using Sentence-Transformers
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 Custom ReAct-inspired agentic retrieval
- 📚 Multi-paper academic literature search
- 🔬 Research-gap identification
- 🧪 Methodology extraction
- 🔄 Cross-paper comparison
- ⚡ FAISS-based vector similarity search
- 🧩 LangChain-based LLM workflow
- 📑 Runtime PDF upload and retrieval
- 📊 Simple RAG vs Agentic RAG evaluation
- 📝 LLM-as-a-Judge evaluation
- 🌐 Streamlit research interface

---

## 🏗️ System Architecture

```mermaid
flowchart LR

    A[Academic PDF Papers]

    A --> B[Phase 1<br/>PDF Ingestion & Chunking]

    B --> C[Phase 2<br/>SBERT Embeddings]

    C --> D[(FAISS<br/>IndexFlatIP)]

    D --> E[Phase 3<br/>Simple RAG]

    D --> F[Phase 4<br/>Agentic RAG]

    E --> G[LLM Response]

    F --> H[Multi-Step Retrieval]

    H --> I[Evidence Synthesis]

    I --> G

    G --> J[Research Answer]

    E --> K[Phase 5<br/>Evaluation]

    F --> K

    K --> L[LLM-as-a-Judge]