# ============================================================
# ResearchForge | Piyush Bhati | SIG 2025-26
# ============================================================
import os
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer as _ST
import faiss as _faiss
print("Heavy libraries pre-loaded OK")

import tempfile, re, json
import numpy as np
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ResearchForge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:#0F1117;color:#E8E8E8;}
[data-testid="stSidebar"]{background:#161B27;border-right:1px solid #2A2F3E;}
.logo-title{font-size:22px;font-weight:700;color:#FFF;letter-spacing:-.5px;}
.logo-sub{font-size:11px;color:#6B7280;margin-top:3px;text-transform:uppercase;letter-spacing:.5px;}
.badge{display:inline-flex;align-items:center;gap:6px;background:#1A2E1A;border:1px solid #2D5A2D;border-radius:20px;padding:4px 12px;font-size:12px;color:#4ADE80;font-weight:500;margin-bottom:8px;}
.lbl{font-size:11px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.hero-title{font-size:38px;font-weight:700;color:#FFF;letter-spacing:-1.5px;line-height:1.1;margin-bottom:12px;}
.hero-title span{color:#6366F1;}
.hero-sub{font-size:15px;color:#9CA3AF;max-width:600px;line-height:1.6;margin-bottom:28px;}
.sh{font-size:11px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin:20px 0 10px 0;}
.abox{background:#161B27;border:1px solid #2A2F3E;border-radius:12px;padding:22px;margin:12px 0;line-height:1.8;color:#E0E0E0;font-size:14px;white-space:pre-wrap;font-family:'Inter',sans-serif;}
.abox-g{background:#0F1F0F;border:1px solid #2D5A2D;border-radius:12px;padding:22px;margin:12px 0;line-height:1.8;color:#E0E0E0;font-size:14px;white-space:pre-wrap;font-family:'Inter',sans-serif;}
.abox-u{background:#1A1628;border:1px solid #4C3B8A;border-radius:12px;padding:22px;margin:12px 0;line-height:1.8;color:#E0E0E0;font-size:14px;white-space:pre-wrap;font-family:'Inter',sans-serif;}
.citem{font-family:'JetBrains Mono',monospace;font-size:12px;color:#9CA3AF;padding:5px 0;border-bottom:1px solid #1E2433;}
.pill{display:inline-block;background:#1A1F2E;border:1px solid #2A3347;border-radius:6px;padding:3px 10px;font-size:12px;color:#9CA3AF;margin:3px;font-family:'JetBrains Mono',monospace;}
.pill-up{display:inline-block;background:#1A1228;border:1px solid #4C3B8A;border-radius:6px;padding:3px 10px;font-size:12px;color:#A78BFA;margin:3px;font-family:'JetBrains Mono',monospace;}
.step{background:#0D1117;border-left:3px solid #6366F1;border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:#9CA3AF;}
.stnum{font-size:11px;color:#6366F1;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;}
.stButton button{background:#1E2433!important;color:#C4C9D4!important;border:1px solid #2A3347!important;border-radius:8px!important;font-size:13px!important;font-weight:500!important;}
.stButton button:hover{background:#252D40!important;border-color:#6366F1!important;color:#FFF!important;}
.stButton button[kind="primary"]{background:#6366F1!important;color:#FFF!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:14px!important;padding:10px 28px!important;}
.stTextInput input{background:#161B27!important;border:1px solid #2A2F3E!important;border-radius:10px!important;color:#E8E8E8!important;font-size:15px!important;padding:12px 16px!important;}
.stTextInput input:focus{border-color:#6366F1!important;box-shadow:0 0 0 2px rgba(99,102,241,.2)!important;}
[data-testid="stMetric"]{background:#161B27;border:1px solid #2A2F3E;border-radius:10px;padding:16px!important;}
[data-testid="stMetricLabel"]{color:#6B7280!important;font-size:11px!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:#FFF!important;font-size:26px!important;font-weight:700!important;}
.stRadio label{color:#C4C9D4!important;}
hr{border-color:#2A2F3E!important;}
</style>
""", unsafe_allow_html=True)


# ── Load base system once (cached) ───────────────────────
@st.cache_resource(show_spinner="Loading ResearchForge…")
def load_base_system():
    """
    Loads and caches: SBERT model, base FAISS index,
    base chunks, Groq client, rag_pipeline, react_agent.
    Called ONCE. Session state extends these at runtime.
    """
    from phase2_embed import load_embedding_model, load_index
    from phase3_rag   import setup_llm
    from phase4_agent import react_agent
    model         = load_embedding_model()
    index, chunks = load_index()
    client        = setup_llm()
    return model, index, chunks, client, react_agent

try:
    model, base_index, base_chunks, client, react_agent = \
        load_base_system()
except Exception as e:
    st.error(f"❌ Loading failed: {e}")
    st.stop()


# ── Session state init ────────────────────────────────────
# We keep a LIVE index and chunks in session state.
# Uploads add to these — base system is never mutated.
if "live_index"    not in st.session_state:
    # Copy base index into session state so uploads can extend it
    st.session_state.live_index  = base_index
    st.session_state.live_chunks = list(base_chunks)
    st.session_state.uploaded    = []   # names of uploaded PDFs
    st.session_state.history     = []
    st.session_state.q           = ""
    st.session_state.go          = False


# ── RAG search — ALWAYS uses live session state ───────────
def live_rag_search(question, top_k=5, uploaded_only=False):
    """
    THE FIX: This function ALWAYS searches st.session_state
    live_index and live_chunks — never the stale base variables.

    uploaded_only=True → filter chunks to uploaded PDFs only
    uploaded_only=False → search everything (base + uploaded)

    This is the core fix for "uploaded PDF not answering":
    Previously rag_pipeline rebuilt its own temp index which
    had a double-normalisation bug. Now we search the already-
    correct live_index directly.
    """
    from phase2_embed import search_faiss

    ci = st.session_state.live_index
    cc = st.session_state.live_chunks

    if uploaded_only and st.session_state.uploaded:
        # Filter chunks to uploaded PDFs only
        up_names = set(st.session_state.uploaded)
        cc_filtered = [c for c in cc if c['source'] in up_names]
        if not cc_filtered:
            return []
        # Build a fresh index from uploaded chunks only
        import faiss, numpy as np
        texts = [c['text'] for c in cc_filtered]
        embs  = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True   # normalise ONCE here
        ).astype(np.float32)
        # Do NOT call faiss.normalize_L2 again — already normalised
        tmp_idx = _faiss.IndexFlatIP(384)
        tmp_idx.add(embs)
        q_emb = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        n = min(top_k, len(cc_filtered))
        dists, idxs = tmp_idx.search(q_emb, n)
        results = []
        for i, idx in enumerate(idxs[0]):
            if 0 <= idx < len(cc_filtered):
                results.append({
                    "text"  : cc_filtered[idx]['text'],
                    "source": cc_filtered[idx]['source'],
                    "score" : float(dists[0][i])
                })
        return results
    else:
        # Search everything — use live index directly
        return search_faiss(question, model, ci, cc, top_k=top_k)


def live_generate_answer(question, chunks_used,
                          uploaded_priority=False):
    """Generates answer using Groq. No hallucinated RAG definition."""
    from phase3_rag import build_prompt, generate_answer
    up_names = (st.session_state.uploaded
                if uploaded_priority else None)
    return generate_answer(question, chunks_used, client,
                           uploaded_sources=up_names)


def live_rag_pipeline(question, top_k=5, scope="all"):
    """
    Clean RAG pipeline that uses session state correctly.
    scope: "uploaded" → search uploaded PDF only
           "all"      → search everything
    """
    uploaded_only = (scope == "uploaded" and
                     bool(st.session_state.uploaded))

    relevant_chunks = live_rag_search(
        question, top_k=top_k, uploaded_only=uploaded_only
    )

    if not relevant_chunks:
        # Fallback to full search if uploaded search empty
        relevant_chunks = live_rag_search(question, top_k=top_k)

    sources = list(set(r['source'] for r in relevant_chunks))
    answer  = live_generate_answer(
        question, relevant_chunks,
        uploaded_priority=uploaded_only
    )

    return {
        "question"    : question,
        "answer"      : answer,
        "sources"     : sources,
        "chunks_used" : relevant_chunks,
        "scope"       : scope
    }


# ── PDF upload processor ──────────────────────────────────
def process_and_add_pdf(uf):
    """
    Reads uploaded PDF, chunks it, embeds it, and adds
    to st.session_state.live_index and live_chunks.
    Returns number of chunks added.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uf.read())
        tmp_path = tmp.name

    text = ""
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        os.unlink(tmp_path)
        return 0, str(e)
    os.unlink(tmp_path)

    if not text.strip():
        return 0, "No readable text found in PDF (may be scanned/image-based)"

    text = re.sub(r' +',    ' ',    text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    words  = text.split()
    chunks = []
    for i in range(0, len(words), 450):
        cw = words[i:i+500]
        if len(cw) >= 80:
            chunks.append({
                "chunk_id": f"up_{uf.name}_{i}",
                "source"  : uf.name,
                "text"    : ' '.join(cw)
            })

    if not chunks:
        return 0, "PDF too short — no chunks created"

    # Embed chunks
    texts = [c['text'] for c in chunks]
    embs  = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True   # normalise ONCE — correct
    ).astype(np.float32)
    # Do NOT call faiss.normalize_L2 here — already normalised above

    # Add to live session state index + chunks
    st.session_state.live_index.add(embs)
    st.session_state.live_chunks.extend(chunks)
    st.session_state.uploaded.append(uf.name)

    return len(chunks), None


# ── Structured prompts ────────────────────────────────────
def gap_prompt(q):
    return (
        f"You are an expert academic research analyst.\n\n"
        f"Search the retrieved paper excerpts and find SPECIFIC "
        f"research gaps about: {q}\n\n"
        f"For EACH paper that mentions limitations or future work:\n\n"
        f"PAPER: [exact filename]\n"
        f"LIMITATION: [specific limitation in that paper]\n"
        f"FUTURE WORK: [what authors suggest next]\n"
        f"---\n\n"
        f"After all papers add:\n\n"
        f"OVERALL RESEARCH DIRECTION:\n"
        f"[3-4 bullet points of most important unsolved problems]\n\n"
        f"RULES:\n"
        f"- Only report gaps ACTUALLY stated in the papers\n"
        f"- Do NOT invent gaps\n"
        f"- Analyse minimum 2 papers"
    )

def method_prompt(q):
    return (
        f"You are an expert academic research analyst.\n\n"
        f"The user is asking about: {q}\n\n"
        f"For EACH paper found write:\n\n"
        f"PAPER: [paper filename]\n"
        f"METHOD: [specific technique or framework used]\n"
        f"DATASET: [data used — write Not specified if absent]\n"
        f"RESULTS: [key findings with numbers if available]\n"
        f"WHY CHOSEN: [reason authors chose this approach]\n"
        f"---\n\n"
        f"IMPORTANT RULE FOR COMPARISON SECTION:\n"
        f"- Count how many unique papers were found\n"
        f"- If only ONE paper was found, write exactly:\n"
        f"  'Only one paper matched the query. "
        f"Cross-paper comparison is not applicable.'\n"
        f"- If TWO OR MORE papers found, write a comparison table:\n"
        f"  | Paper | Method | Key Result |\n"
        f"  |-------|--------|------------|\n"
        f"  [one row per paper]\n\n"
        f"STRICT RULES:\n"
        f"- Only use information from the retrieved excerpts\n"
        f"- Include numbers and metrics wherever papers provide them\n"
        f"- Never invent a second paper if only one was retrieved\n"
        f"- Never write 'Paper: Not specified'"
    )

def compare_prompt(q):
    return (
        f"You are an expert academic research analyst.\n\n"
        f"The user wants to compare papers on: {q}\n\n"
        f"FIRST — count how many unique papers are in the evidence.\n\n"
        f"IF ONLY ONE PAPER IS FOUND:\n"
        f"Write exactly this structure:\n\n"
        f"Only one relevant paper was retrieved.\n"
        f"Paper comparison requires at least two papers.\n\n"
        f"Summary of retrieved paper:\n"
        f"PAPER: [filename]\n"
        f"MAIN CLAIM: [what the paper argues]\n"
        f"KEY RESULT: [most important finding with numbers]\n"
        f"STRENGTH: [what this paper does well]\n"
        f"LIMITATION: [what this paper is missing]\n\n"
        f"IF TWO OR MORE PAPERS ARE FOUND:\n"
        f"Write this structure for each paper:\n\n"
        f"PAPER: [filename]\n"
        f"MAIN CLAIM: [what the paper argues]\n"
        f"KEY RESULT: [most important finding with numbers]\n"
        f"STRENGTH: [what this paper does well]\n"
        f"WEAKNESS: [what this paper is missing]\n"
        f"---\n\n"
        f"Then add:\n\n"
        f"KEY DIFFERENCES:\n"
        f"[bullet points on how these papers differ]\n\n"
        f"CONSENSUS:\n"
        f"[What ALL papers agree on]\n\n"
        f"STRICT RULES:\n"
        f"- Count papers BEFORE writing comparison\n"
        f"- Never compare a paper with itself\n"
        f"- Never invent a second paper\n"
        f"- Be specific with paper names and numbers"
    )


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 10px 0;border-bottom:1px solid
    #2A2F3E;margin-bottom:16px;">
    <div class="logo-title">🔬 ResearchForge</div>
    <div class="logo-sub">Piyush Bhati | SIG 2025-26</div></div>
    """, unsafe_allow_html=True)

    _lc  = st.session_state.live_chunks
    _lp  = len(set(c['source'] for c in _lc))
    _up  = len(st.session_state.uploaded)
    st.markdown(
        f'<div class="badge">● Online — '
        f'{len(_lc)} chunks · {_lp} papers'
        f'{f" · {_up} uploaded" if _up else ""}</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="lbl">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", [
        "⚡ Quick Answer",
        "🧠 Deep Research",
        "🔬 Research Gap Finder",
        "📋 Methodology Finder",
        "📊 Compare Papers",
        "⚖️ Compare RAG vs Agent",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="lbl">Upload PDF</div>',
                unsafe_allow_html=True)

    # Search scope — only shown when PDFs are uploaded
    if st.session_state.uploaded:
        scope = st.radio("Search scope", [
            "📄 Uploaded PDF only",
            "📚 All papers (base + uploaded)"
        ], label_visibility="visible", key="scope")
        search_scope = ("uploaded"
                        if "Uploaded" in scope else "all")
    else:
        search_scope = "all"

    ufs = st.file_uploader(
        "Add a PDF to your knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if ufs:
        new_files = [f for f in ufs
                     if f.name not in st.session_state.uploaded]
        if new_files:
            if st.button(f"＋ Index {len(new_files)} PDF(s)",
                         key="addbtn"):
                for uf in new_files:
                    with st.spinner(f"Reading & embedding {uf.name}…"):
                        n, err = process_and_add_pdf(uf)
                    if err:
                        st.warning(f"⚠ {uf.name}: {err}")
                    else:
                        st.success(
                            f"✅ {uf.name} — {n} passages indexed\n"
                            f"Switch scope to 'Uploaded PDF only' "
                            f"to search it directly."
                        )
                st.rerun()
        else:
            st.caption("All selected PDFs already indexed")

    if st.session_state.uploaded:
        st.markdown('<div class="lbl" style="margin-top:10px">'
                    'Uploaded PDFs</div>', unsafe_allow_html=True)
        for name in st.session_state.uploaded:
            st.markdown(f'<div class="pill-up">📄 {name[:30]}</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.caption(
        f"{len(st.session_state.live_chunks)} passages · "
        f"{len(set(c['source'] for c in st.session_state.live_chunks))} papers"
    )
    if st.session_state.history:
        st.divider()
        st.markdown('<div class="lbl">Recent</div>',
                    unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-5:]):
            st.caption(f"↳ {q[:44]}…")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:36px 0 16px 0;">
<div class="hero-title">🔬 ResearchForge<br>
<span>Agentic AI Research Assistant</span></div>
<div class="hero-sub">Piyush Bhati | M.Sc. Data Science & Spatial Analytics | SIG 2025-26</div></div>
""", unsafe_allow_html=True)

question = st.text_input("",
    value=st.session_state.q,
    placeholder="Ask a research question…",
    label_visibility="collapsed")

st.markdown('<div class="sh">Suggested questions</div>',
            unsafe_allow_html=True)

chips1 = [
    ("What is RAG?",        "What is Retrieval Augmented Generation?"),
    ("How does SBERT work?","How does SBERT improve sentence search over BERT?"),
    ("What is ReAct?",      "What is the ReAct framework and how does it work?"),
    ("What is FAISS?",      "How does FAISS store and search vectors efficiently?"),
]
for col, (lbl, q) in zip(st.columns(4), chips1):
    with col:
        if st.button(lbl, key=f"c1_{lbl}"):
            st.session_state.q  = q
            st.session_state.go = True
            st.rerun()

chips2 = [
    ("RAG + Hallucination",
     "How does RAG reduce hallucination and how is this measured?"),
    ("ReAct vs Simple RAG",
     "Compare ReAct agent vs simple RAG for complex questions"),
    ("Research Gaps",
     "What limitations and future work do these papers identify?"),
    ("Compare Methods",
     "Compare the main methods and findings across all papers"),
]
for col, (lbl, q) in zip(st.columns(4), chips2):
    with col:
        if st.button(lbl, key=f"c2_{lbl}"):
            st.session_state.q  = q
            st.session_state.go = True
            st.rerun()

st.markdown("")
go_btn = st.button("🔍 Search", type="primary")

if st.session_state.go:
    go_btn   = True
    question = st.session_state.q
    st.session_state.go = False


# ── Helper display functions ──────────────────────────────
def show_trace(trace):
    with st.expander("🧠 Agent reasoning trace"):
        for s in trace:
            st.markdown(
                f'<div class="stnum">Step {s["step"]}</div>'
                f'<div class="step"><b>Thought:</b> {s["thought"]}<br>'
                f'<b>Action:</b> <code>{s["action"]}</code><br>'
                f'<b>Found:</b> {s["observation"][:250]}…</div>',
                unsafe_allow_html=True)

def show_sources(sources):
    for src in sources:
        is_up = src in st.session_state.uploaded
        cls   = "pill-up" if is_up else "pill"
        tag   = " · uploaded" if is_up else ""
        st.markdown(f'<div class="{cls}">{src}{tag}</div>',
                    unsafe_allow_html=True)

def show_citations(citations):
    for c in citations:
        st.markdown(f'<div class="citem">→ {c}</div>',
                    unsafe_allow_html=True)

def show_passages(chunks_used):
    with st.expander("View retrieved passages"):
        for i, ch in enumerate(chunks_used):
            is_up = ch['source'] in st.session_state.uploaded
            tag   = " · 📄 UPLOADED" if is_up else ""
            st.markdown(
                f'<div class="stnum">Passage {i+1} · '
                f'score {ch["score"]:.3f} · {ch["source"]}{tag}</div>'
                f'<div class="step">{ch["text"][:400]}…</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════
if go_btn and question:
    if question not in st.session_state.history:
        st.session_state.history.append(question)

    st.divider()

    # Show scope banner if uploaded PDF is active
    if (search_scope == "uploaded" and
            st.session_state.uploaded):
        st.info(
            f"🔍 Searching uploaded PDF(s) only: "
            f"{', '.join(st.session_state.uploaded)}"
        )

    try:

        # ── QUICK ANSWER ──────────────────────────────────
        if "Quick" in mode:
            with st.spinner("Searching…"):
                r = live_rag_pipeline(
                    question, top_k=5, scope=search_scope
                )

            box_cls = "abox-u" if r["scope"] == "uploaded" else "abox"
            st.markdown('<div class="sh">Answer</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="{box_cls}">{r["answer"]}</div>',
                        unsafe_allow_html=True)

            if r["scope"] == "uploaded":
                st.success("✅ Answer from your uploaded PDF")

            st.download_button("↓ Export answer",
                f"Q: {question}\n\nA: {r['answer']}\n\n"
                f"Sources: {', '.join(r['sources'])}",
                file_name="answer.txt")

            ca, cb = st.columns([2,1])
            with ca:
                st.markdown('<div class="sh">Sources</div>',
                            unsafe_allow_html=True)
                show_sources(r['sources'])
            with cb:
                c1, c2 = st.columns(2)
                c1.metric("Passages", len(r['chunks_used']))
                c2.metric("Sources",  len(r['sources']))
            show_passages(r['chunks_used'])

        # ── DEEP RESEARCH ─────────────────────────────────
        elif "Deep" in mode:
            with st.spinner("Agent reasoning across papers… (30-60s)"):
                r = react_agent(
                    question, model,
                    st.session_state.live_index,
                    st.session_state.live_chunks,
                    client, max_searches=3,
                    uploaded_sources=st.session_state.uploaded or None,
                    search_scope=search_scope,
                    force_mode="deep"
                )
            st.markdown('<div class="sh">Answer</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="abox-g">{r["answer"]}</div>',
                        unsafe_allow_html=True)
            st.download_button("↓ Export + Citations",
                f"Q: {question}\n\nA: {r['answer']}\n\n"
                f"Citations:\n" + "\n".join(r['citations']),
                file_name="deep_answer.txt")
            m1, m2, m3 = st.columns(3)
            m1.metric("Searches",  r['steps_taken'])
            m2.metric("Sources",   len(r['sources']))
            m3.metric("Citations", len(r['citations']))
            st.markdown('<div class="sh">Citations</div>',
                        unsafe_allow_html=True)
            show_citations(r['citations'])
            show_trace(r['reasoning_trace'])

        # ── RESEARCH GAP FINDER ───────────────────────────
        elif "Gap" in mode:
            st.info("🔬 Finding limitations and future work… (45-60s)")
            with st.spinner("Searching for research gaps…"):
                r = react_agent(
                    gap_prompt(question), model,
                    st.session_state.live_index,
                    st.session_state.live_chunks,
                    client, max_searches=3,
                    uploaded_sources=st.session_state.uploaded or None,
                    search_scope=search_scope,
                    force_mode="gap"
                )
            st.markdown('<div class="sh">Research Gaps</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="abox">{r["answer"]}</div>',
                        unsafe_allow_html=True)
            st.download_button("↓ Export gap analysis",
                f"RESEARCH GAP ANALYSIS\nTopic: {question}\n\n"
                f"{r['answer']}\n\nPapers: "
                f"{', '.join(r['sources'])}",
                file_name="gap_analysis.txt")
            m1, m2 = st.columns(2)
            m1.metric("Papers Analysed", len(r['sources']))
            m2.metric("Search Rounds",   r['steps_taken'])
            st.markdown('<div class="sh">Papers Searched</div>',
                        unsafe_allow_html=True)
            show_sources(r['sources'])
            show_trace(r['reasoning_trace'])

        # ── METHODOLOGY FINDER ────────────────────────────
        elif "Methodology" in mode:
            st.info("📋 Extracting methodology… (45-60s)")
            with st.spinner("Extracting methodology…"):
                r = react_agent(
                    method_prompt(question), model,
                    st.session_state.live_index,
                    st.session_state.live_chunks,
                    client, max_searches=3,
                    uploaded_sources=st.session_state.uploaded or None,
                    search_scope=search_scope,
                    force_mode="methodology"
                )
            st.markdown('<div class="sh">Methodology</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="abox-g">{r["answer"]}</div>',
                        unsafe_allow_html=True)
            st.download_button("↓ Export methodology",
                f"METHODOLOGY\nTopic: {question}\n\n"
                f"{r['answer']}\n\nSources: "
                f"{', '.join(r['sources'])}",
                file_name="methodology.txt")
            st.markdown('<div class="sh">Citations</div>',
                        unsafe_allow_html=True)
            show_citations(r['citations'])
            show_trace(r['reasoning_trace'])

        # ── COMPARE PAPERS ────────────────────────────────
        elif "Compare Papers" in mode:
            st.info("📊 Comparing papers… (45-60s)")
            # Force compare keyword so phase4 keyword detection works
            # Add compare prefix only if not already there
            forced_q = question if question.lower().startswith("compare") else f"compare {question}"
            with st.spinner("Comparing across papers…"):
                r = react_agent(
                    compare_prompt(forced_q), model,
                    st.session_state.live_index,
                    st.session_state.live_chunks,
                    client, max_searches=3,
                    uploaded_sources=st.session_state.uploaded or None,
                    search_scope=search_scope,
                    force_mode="comparison"
                )
            st.markdown('<div class="sh">Paper Comparison</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="abox">{r["answer"]}</div>',
                        unsafe_allow_html=True)
            st.download_button("↓ Export comparison",
                f"PAPER COMPARISON\nTopic: {question}\n\n"
                f"{r['answer']}\n\nSources: "
                f"{', '.join(r['sources'])}",
                file_name="paper_comparison.txt")
            m1, m2 = st.columns(2)
            m1.metric("Papers Compared", len(r['sources']))
            m2.metric("Search Rounds",   r['steps_taken'])
            show_sources(r['sources'])
            show_trace(r['reasoning_trace'])

        # ── COMPARE RAG vs AGENT ──────────────────────────
        elif "Compare RAG" in mode:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("""
                <div style='background:#1A1F2E;border:1px solid #2A3347;
                border-radius:10px;padding:12px 16px;margin-bottom:10px;'>
                <span style='font-size:11px;font-weight:600;color:#9CA3AF;
                text-transform:uppercase;'>⚡ Quick Answer</span><br>
                <span style='font-size:11px;color:#6B7280;'>
                1 search · Fast</span></div>""",
                             unsafe_allow_html=True)
                with st.spinner("Searching…"):
                    rs = live_rag_pipeline(
                        question, top_k=5, scope=search_scope
                    )
                st.markdown(f'<div class="abox">{rs["answer"]}</div>',
                            unsafe_allow_html=True)
                show_sources(rs['sources'])
                x1, x2 = st.columns(2)
                x1.metric("Searches", 1)
                x2.metric("Passages", len(rs['chunks_used']))

            with col_r:
                st.markdown("""
                <div style='background:#1A2E1A;border:1px solid #2D5A2D;
                border-radius:10px;padding:12px 16px;margin-bottom:10px;'>
                <span style='font-size:11px;font-weight:600;color:#4ADE80;
                text-transform:uppercase;'>🧠 Deep Research</span><br>
                <span style='font-size:11px;color:#6B7280;'>
                3 searches · Cited</span></div>""",
                             unsafe_allow_html=True)
                with st.spinner("Agent reasoning…"):
                    ra = react_agent(
                        question, model,
                        st.session_state.live_index,
                        st.session_state.live_chunks,
                        client, max_searches=3,
                        uploaded_sources=st.session_state.uploaded or None,
                        search_scope=search_scope
                    )
                st.markdown(f'<div class="abox-g">{ra["answer"]}</div>',
                            unsafe_allow_html=True)
                show_citations(ra['citations'])
                x3, x4 = st.columns(2)
                x3.metric("Searches",  ra['steps_taken'])
                x4.metric("Citations", len(ra['citations']))

            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Simple Sources",  len(rs['sources']))
            s2.metric("Agent Sources",   len(ra['sources']))
            s3.metric("Agent Steps",     ra['steps_taken'])
            s4.metric("Agent Citations", len(ra['citations']))
            st.markdown("""
            <div class="abox" style="margin-top:14px;">
            <b style="color:#6366F1;">Key difference:</b>
            Quick Answer does <b>1 search</b> → returns closest passage.
            Deep Research <b>plans 3 searches</b> → synthesises all
            findings → cites every claim → better for complex questions.
            </div>""", unsafe_allow_html=True)
            st.download_button("↓ Export comparison",
                f"Q: {question}\n\nQUICK:\n{rs['answer']}\n"
                f"Sources: {', '.join(rs['sources'])}\n\n"
                f"DEEP:\n{ra['answer']}\n"
                f"Citations:\n" + "\n".join(ra['citations']),
                file_name="comparison.txt")
            show_trace(ra['reasoning_trace'])

    except Exception as e:
        st.error(f"Error: {e}")
        st.caption("Try Quick Answer mode or rephrase your question.")

elif go_btn and not question:
    st.warning("Please enter a question.")