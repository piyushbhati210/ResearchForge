# ============================================================
# PHASE 3 — RAG Pipeline with Upload Priority
# ResearchForge | Piyush Bhati | SIG 2025-26
# ============================================================
# IMPROVEMENT (ChatGPT suggestion July 2026):
# When user uploads a PDF, search uploaded chunks FIRST.
# If enough evidence found → answer from uploaded PDF.
# If not enough → fallback to full knowledge base.
# This makes uploaded PDFs actually answer questions.
# ============================================================

import os
import json
from dotenv import load_dotenv

load_dotenv()


# ── FUNCTION 1: Setup Groq LLM ───────────────────────────
def setup_llm():
    """
    Connects to Groq API using key from .env file.
    WHY Groq? Free, fast, runs LLaMA 3.1 with no GPU needed.
    """
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file")
        exit(1)
    client = Groq(api_key=api_key)
    print("Groq API connected (LLaMA 3.1)")
    return client


# ── FUNCTION 2: Build RAG prompt ─────────────────────────
def build_prompt(question, context_chunks,
                 uploaded_sources=None):
    """
    Builds the RAG prompt.
    If uploaded_sources provided, adds a note to prioritise them.
    """
    context = ""
    for i, chunk in enumerate(context_chunks):
        words     = chunk['text'].split()[:200]
        truncated = ' '.join(words)
        # Mark uploaded chunks clearly
        is_uploaded = (
            uploaded_sources and
            chunk['source'] in uploaded_sources
        )
        tag = " [UPLOADED BY USER]" if is_uploaded else ""
        context += f"\n[Paper {i+1}: {chunk['source']}{tag}]\n"
        context += truncated
        context += "\n"

    # Add priority instruction if user uploaded something
    upload_note = ""
    if uploaded_sources:
        names = ", ".join(uploaded_sources)
        upload_note = (
            f"\nNOTE: The user uploaded: {names}\n"
            f"Prioritise information from uploaded papers "
            f"marked [UPLOADED BY USER] in your answer.\n"
        )

    prompt = f"""You are an expert academic research assistant.
RAG ALWAYS means Retrieval-Augmented Generation (Lewis et al. 2020).
{upload_note}
RETRIEVED PAPER EXCERPTS:
{context}

QUESTION: {question}

STEP 1 — Scan ALL excerpts carefully for:
- The main research question or objective of the paper
- Specific numbers, percentages, coefficients
- Direct statements of findings or results
- Named datasets, authors, methods
- Concrete facts that directly answer the question

STEP 2 — Write your answer following these rules:
- Answer in maximum 4 sentences
- Lead with the direct answer immediately — no preamble
- For "research question" queries: state what the paper examines and its main finding
- Include specific numbers from the paper (e.g. "1 percent increase")
- Cite ONLY the paper name once at the end, not before every sentence
- If [UPLOADED BY USER] paper has the answer, use it FIRST
- Do NOT repeat the same point from the same paper multiple times
- Do NOT cite "Paper 1", "Paper 2" — use the actual filename
- Do NOT add "Additional context" or "Not in the provided papers" sections
- If a specific number exists in the excerpts, you MUST include it
- Never cut off mid-sentence
- Keep answer focused — maximum 4 sentences, no padding

ANSWER:"""
    return prompt


# ── FUNCTION 3: Generate answer ───────────────────────────
def generate_answer(question, context_chunks, client,
                    uploaded_sources=None):
    """
    Sends prompt to Groq LLaMA 3.1 and returns the answer.
    """
    prompt = build_prompt(question, context_chunks,
                          uploaded_sources)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()


# ── FUNCTION 4: Search uploaded chunks first ─────────────
def search_uploaded_first(question, model,
                           all_chunks, uploaded_names,
                           top_k=5):
    """
    NEW: Searches uploaded chunks with priority.

    Strategy (ChatGPT suggestion):
    1. Filter chunks belonging to uploaded PDFs
    2. Search those first (if any uploaded)
    3. If uploaded chunks score well → use them
    4. Fill remaining slots from full knowledge base
    5. Always return top_k total chunks

    WHY this matters:
    Without priority, 653 existing chunks dominate results.
    Uploaded PDF (18 chunks) rarely ranks in top 5.
    With priority, uploaded chunks are always considered first.
    """
    from phase2_embed import search_faiss

    # Separate uploaded vs base chunks
    uploaded_chunks = [
        c for c in all_chunks
        if c['source'] in uploaded_names
    ]
    base_chunks = [
        c for c in all_chunks
        if c['source'] not in uploaded_names
    ]

    if not uploaded_chunks:
        # No uploaded files — normal search
        return search_faiss(
            question, model,
            None, all_chunks, top_k=top_k
        ), []

    # Search uploaded chunks first
    from phase2_embed import load_index
    import faiss
    import numpy as np

    # Build temporary index for uploaded chunks only
    model_dim = 384
    temp_index = faiss.IndexFlatIP(model_dim)

    up_texts = [c['text'] for c in uploaded_chunks]
    up_embs  = model.encode(
        up_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True   # normalise ONCE only
    ).astype(np.float32)
    # Do NOT call faiss.normalize_L2 — already normalised above
    temp_index.add(up_embs)

    # Search uploaded index
    q_emb = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True   # normalise ONCE only
    ).astype(np.float32)
    # Do NOT call faiss.normalize_L2 — already normalised above

    n_up  = min(top_k, len(uploaded_chunks))
    dists, idxs = temp_index.search(q_emb, n_up)

    uploaded_results = []
    for i, idx in enumerate(idxs[0]):
        if idx < len(uploaded_chunks):
            uploaded_results.append({
                "text"  : uploaded_chunks[idx]['text'],
                "source": uploaded_chunks[idx]['source'],
                "score" : float(dists[0][i])
            })

    # Fill remaining slots from full knowledge base
    remaining = top_k - len(uploaded_results)
    if remaining > 0:
        base_results = search_faiss(
            question, model,
            None, all_chunks,
            top_k=top_k + len(uploaded_chunks)
        )
        # Exclude already-found uploaded results
        up_sources = set(r['source'] for r in uploaded_results)
        base_filtered = [
            r for r in base_results
            if r['source'] not in up_sources
        ][:remaining]
        combined = uploaded_results + base_filtered
    else:
        combined = uploaded_results

    return combined, uploaded_names


# ── FUNCTION 5: Full RAG pipeline ────────────────────────
def rag_pipeline(question, model, index, chunks,
                 client, top_k=5,
                 uploaded_sources=None):
    """
    Complete RAG pipeline with upload priority.

    If uploaded_sources provided:
    → Search uploaded chunks first (priority mode)
    → Fill remaining with base knowledge base
    → Mark uploaded sources in prompt

    If no uploaded_sources:
    → Normal FAISS search over everything
    """
    from phase2_embed import search_faiss

    print(f"\nQuestion: {question}")

    if uploaded_sources and len(uploaded_sources) > 0:
        # Priority mode: uploaded PDF searched first
        print(f"Priority search: {uploaded_sources}")
        relevant_chunks, up_names = search_uploaded_first(
            question, model, chunks,
            uploaded_sources, top_k=top_k
        )
        print(f"Sources found: "
              f"{list(set(r['source'] for r in relevant_chunks))}")
    else:
        # Normal mode: search everything equally
        print("Searching all papers...")
        relevant_chunks = search_faiss(
            question, model, index, chunks, top_k=top_k
        )
        up_names = None

    sources = list(set([r["source"] for r in relevant_chunks]))
    print(f"Generating answer from: {sources}")

    answer = generate_answer(
        question, relevant_chunks, client,
        uploaded_sources=up_names
    )

    return {
        "question"        : question,
        "answer"          : answer,
        "sources"         : sources,
        "chunks_used"     : relevant_chunks,
        "uploaded_priority": bool(uploaded_sources)
    }


# ── MAIN: Test Phase 3 ────────────────────────────────────
if __name__ == "__main__":
    from phase2_embed import load_embedding_model, load_index

    model         = load_embedding_model()
    index, chunks = load_index()
    client        = setup_llm()

    test_questions = [
        "What is Retrieval Augmented Generation?",
        "How does RAG reduce hallucination in LLMs?",
        "How does the ReAct framework work?",
        "What metrics does RAGAS use to evaluate RAG?",
        "How does SBERT improve sentence search?",
    ]

    os.makedirs("outputs", exist_ok=True)
    all_results = []

    print("\n" + "="*60)
    print("TESTING RAG PIPELINE WITH UPLOAD PRIORITY")
    print("="*60)

    for question in test_questions:
        result = rag_pipeline(
            question, model, index, chunks, client
        )
        print(f"\nQ: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print("-" * 60)
        all_results.append({
            "question": result["question"],
            "answer"  : result["answer"],
            "sources" : result["sources"]
        })

    with open("outputs/rag_results.json", "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to outputs/rag_results.json")
    print("\nPhase 3 Complete!")