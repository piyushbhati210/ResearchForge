# ============================================================
# PHASE 2 — Create Vector Embeddings and FAISS Database
# ResearchForge | Piyush Bhati | SIG 2025-26
# ============================================================
# WHY SentenceTransformer?
# SBERT (Reimers & Gurevych, 2019) converts text into 384-dim
# vectors where similar sentences are close together in space.
# This enables semantic similarity search — finding chunks that
# MEAN the same thing as a query, not just share keywords.
# WHY FAISS?
# FAISS (Johnson et al., 2021) stores millions of vectors and
# finds the most similar ones in milliseconds using approximate
# or exact nearest-neighbour search.
# WHY numpy?
# FAISS works with numpy float32 arrays for fast matrix operations.

import json
import pickle
import os


# ── FUNCTION 1: local_files_only=Trueoad SBERT model ─────────────────────────
def load_embedding_model():
    """
    Loads Sentence-BERT all-MiniLM-L6-v2.

    WHY all-MiniLM-L6-v2?
    384 dimensions — small enough for fast search, large enough
    for strong semantic accuracy. Best balance for RAG systems.

    WHY local_files_only=True?
    Prevents HuggingFace network calls which hang on restricted
    college/hostel networks. Model is cached after first download.

    WHY get_embedding_dimension()?
    Confirms the model loaded correctly. Prints 384 if working.
    """
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT embedding model...")
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        local_files_only=False
    )
    print("Model loaded successfully!")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    return model


# ── FUNCTION 2: Encode chunks into vectors ───────────────
def encode_chunks(chunks, model):
    """
    Converts all text chunks into 384-dimensional vectors.

    WHY normalize_embeddings=True?
    Normalises all vectors to unit length here so we can use
    Inner Product (IndexFlatIP) as cosine similarity in FAISS.
    # Do NOT call faiss.normalize_L2 — already normalised
    double-normalise and reduce vectors near to zero.

    WHY batch_size=32?
    Processes 32 chunks at once — GPU or CPU parallel processing.
    Much faster than encoding one chunk at a time.

    WHY show_progress_bar=True?
    With 653+ chunks this takes 1-3 minutes. The progress bar
    confirms the process is running and not hung.
    """
    import numpy as np
    print(f"\nEncoding {len(chunks)} chunks into vectors...")
    print("This takes 2-5 minutes depending on your computer...")

    texts = [chunk['text'] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # ← normalise HERE, not in FAISS
    )

    embeddings = embeddings.astype('float32')

    print(f"\nEncoding complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Each chunk → {embeddings.shape[1]}-dimensional vector")
    return embeddings


# ── FUNCTION 3: Build FAISS index ────────────────────────
def build_faiss_index(embeddings):
    """
    Creates a FAISS vector database from embeddings.

    WHY IndexFlatIP?
    IP = Inner Product. When vectors are normalised to unit length,
    Inner Product equals Cosine Similarity. This gives us exact
    semantic similarity search with no approximation error.

    WHY NOT IndexIVFFlat or HNSW?
    Those are approximate methods for 1M+ vectors. Our corpus
    has ~653 chunks — exact search is fast enough and more accurate.

    # Do NOT call faiss.normalize_L2 — already normalised
    Vectors are already normalised in encode_chunks.
    Double-normalising collapses all vectors to near-zero magnitude,
    breaking cosine similarity scores completely.
    """
    import faiss
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]     # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, {dimension} dimensions")
    return index


# ── FUNCTION 4: Search FAISS index ───────────────────────
def search_faiss(query, model, index, chunks, top_k=5):
    """
    Given a natural language query, finds the most semantically
    similar chunks from the indexed research papers.

    WHY encode the query with the same model?
    Both query and document chunks must be in the same vector
    space for cosine similarity to be meaningful.

    WHY normalize_embeddings=True for query?
    Query must be unit-length to match the IndexFlatIP index.
    If query is not normalised, Inner Product ≠ Cosine Similarity.

    WHY safety check idx < len(chunks)?
    FAISS can return -1 as an index when fewer than top_k results
    exist. This check prevents IndexError on small corpora.
    """
    import numpy as np

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True   # ← must match index normalisation
    ).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):      # safety check: -1 or out-of-range
            results.append({
                "text"  : chunks[idx]['text'],
                "source": chunks[idx]['source'],
                "score" : float(distances[0][i])
            })
    return results


# ── FUNCTION 5: Save FAISS index ─────────────────────────
def save_index(index, chunks,
               index_path="vectorstore/faiss.index",
               chunks_path="vectorstore/chunks.pkl"):
    """
    Saves FAISS index and chunks list to disk.

    WHY save separately?
    FAISS index stores only the numeric vectors (fast, compact).
    Chunks pickle stores the original text + metadata (source filename).
    Both are needed together to return readable results.

    WHY pickle for chunks?
    JSON cannot store all Python dict structures reliably.
    Pickle is faster and preserves exact data types.

    After saving: running load_index() takes < 1 second instead
    of the 2-5 minute rebuild.
    """
    import faiss
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, index_path)

    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f"\nSaved FAISS index → {index_path}")
    print(f"Saved chunks      → {chunks_path}")
    print(f"Total vectors saved: {index.ntotal}")


# ── FUNCTION 6: Load saved index ─────────────────────────
def load_index(index_path="vectorstore/faiss.index",
               chunks_path="vectorstore/chunks.pkl"):
    """
    Loads the previously saved FAISS index from disk.
    Takes < 1 second. Called by Phase 3, Phase 4, and app.py.

    WHY check file exists?
    If Phase 2 has not been run yet, gives a clear error message
    instead of a confusing FileNotFoundError traceback.
    """
    import faiss

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            f"Please run phase2_embed.py first."
        )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            f"Please run phase2_embed.py first."
        )

    index = faiss.read_index(index_path)

    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    print(f"Loaded FAISS index: {index.ntotal} vectors")
    print(f"Loaded chunks: {len(chunks)} text segments")
    return index, chunks


# ── MAIN: Run Phase 2 ─────────────────────────────────────
if __name__ == "__main__":

    # Load chunks created by Phase 1
    chunks_path = "outputs/chunks.json"
    if not os.path.exists(chunks_path):
        print(f"ERROR: {chunks_path} not found.")
        print("Please run phase1_load.py first.")
        exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from Phase 1")
    print(f"Sample sources: {list(set(c['source'] for c in chunks))[:3]}")

    # Build and save the FAISS index
    model      = load_embedding_model()
    embeddings = encode_chunks(chunks, model)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks)

    # Test search with 3 sample queries
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)

    test_queries = [
        "What is Retrieval Augmented Generation?",
        "How does SBERT improve semantic search?",
        "What is the ReAct framework?"
    ]

    for test_query in test_queries:
        print(f"\nQuery: {test_query}")
        results = search_faiss(
            test_query, model, index, chunks, top_k=3
        )
        for i, r in enumerate(results):
            print(f"  [{i+1}] Score: {r['score']:.4f} | "
                  f"Source: {r['source']}")
            print(f"       Text: {r['text'][:120]}...")

    print("\n" + "="*60)
    print("Phase 2 Complete!")
    print("FAISS index saved to vectorstore/")
    print("Run phase3_rag.py to test question answering.")
    print("="*60)