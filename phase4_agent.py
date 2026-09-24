# ============================================================
# PHASE 4 — Agentic Framework using LangChain + ReAct
# ResearchForge | Piyush Bhati | SIG 2025-26
# ============================================================
# WHY LangChain + ReAct?
# Yao et al. (2023) showed that interleaving Thought → Action
# → Observation loops allows LLMs to handle multi-hop questions
# that single-step RAG cannot answer. LangChain provides the
# ChatGroq wrapper and message types that connect our agent
# to the Groq API using the standard LangChain interface.

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

# Global state shared across tool functions
# Updated fresh on every react_agent call
_model            = None
_index            = None
_chunks           = None
_client           = None
_llm              = None
_uploaded_sources = None  # tracks uploaded PDFs for priority
_search_scope     = "all"  # "uploaded" or "all" — synced from UI
# NOTE: relevance filtering is relative (see ABS_FLOOR/REL_CUTOFF
# inside react_agent's search loop) rather than a fixed absolute
# score, since SBERT similarity scales vary across corpora.


# ── LangChain LLM setup ───────────────────────────────────
def get_langchain_llm():
    """
    Creates a LangChain ChatGroq LLM instance.
    WHY ChatGroq? It is the official LangChain wrapper for Groq,
    giving us standard .invoke() interface used across LangChain.
    WHY max_tokens=600? Synthesis answers need more space than
    simple RAG answers — 600 allows structured formatted output.
    """
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.6,
        max_completion_tokens=4096,
        reasoning_effort="low"
    )
    return llm


def langchain_invoke(llm, prompt_text, max_retries=3):
    """
    Invokes the LangChain LLM with a system + human message pair.
    WHY SystemMessage + HumanMessage?
    This is the standard LangChain message format. SystemMessage
    sets the agent's persona and rules. HumanMessage is the task.
    Using both gives the LLM clear role separation.

    FIX: added retry/backoff for Groq 429 rate-limit errors.
    Groq's free tier has a tokens-per-minute (TPM) cap; running
    several react_agent calls back-to-back (e.g. in the Phase 4/5
    test loops) can exceed it mid-run. Groq's error message includes
    the exact wait time ("Please try again in 2.24s") — we parse
    that and wait accordingly instead of crashing the whole run.
    """
    import re as _re
    import time as _time
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=(
            "You are an expert AI research assistant specialising in academic literature. "
            "RAG ALWAYS means Retrieval-Augmented Generation (Lewis et al. 2020). "
            "Always cite which paper each point comes from. "
            "Always complete every sentence fully — never cut off mid-sentence. "
            "Be specific, technical, and cite paper names for every claim."
        )),
        HumanMessage(content=prompt_text)
    ]

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < max_retries - 1:
                # Try to read Groq's suggested wait time, else default
                wait_match = _re.search(r"try again in ([\d.]+)s", err)
                wait_s = float(wait_match.group(1)) + 2 if wait_match else 15
                print(f"    ⚠️ Groq rate limit hit — waiting {wait_s:.1f}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                _time.sleep(wait_s)
            elif attempt < max_retries - 1:
                print(f"    ⚠️ LLM call failed: {err[:80]} "
                      f"(attempt {attempt+1}/{max_retries}) — retrying in 5s")
                _time.sleep(5)
            else:
                print(f"    ❌ LLM call failed after {max_retries} attempts: {err[:120]}")
                raise


# ── Tool 1: Search papers via RAG ────────────────────────
def search_papers_tool(query):
    """
    LangChain Tool 1: Searches papers respecting UI search scope.

    FIX 1 (GPT): Respects _search_scope — if UI says
    "Uploaded PDF only", agent searches ONLY uploaded chunks.
    Previously agent always searched everything regardless of UI scope.

    FIX 2 (GPT): Removed redundant generate_answer() call.
    The ReAct synthesis already reasons over raw chunks at the end.
    Removing intermediate LLM call saves latency and tokens.

    FIX 3: Source order preserved using loop not set().
    """
    from phase2_embed import search_faiss
    import faiss as _faiss_local
    import numpy as np

    # Respect the UI search scope
    if _search_scope == "uploaded" and _uploaded_sources:
        up_names    = set(_uploaded_sources)
        cc_filtered = [c for c in _chunks if c["source"] in up_names]

        if cc_filtered:
            # Embed and search uploaded chunks only
            texts = [c["text"] for c in cc_filtered]
            embs  = _model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            tmp_idx = _faiss_local.IndexFlatIP(384)
            tmp_idx.add(embs)

            q_emb = _model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)

            n = min(5, len(cc_filtered))
            dists, idxs = tmp_idx.search(q_emb, n)

            chunks_found = []
            for i, idx in enumerate(idxs[0]):
                if 0 <= idx < len(cc_filtered):
                    chunks_found.append({
                        "text"  : cc_filtered[idx]["text"],
                        "source": cc_filtered[idx]["source"],
                        "score" : float(dists[0][i])
                    })
        else:
            chunks_found = search_faiss(
                query, _model, _index, _chunks, top_k=10
            )
    else:
        # Search full live index — base + uploaded
        chunks_found = search_faiss(
            query, _model, _index, _chunks, top_k=10
        )

    # Preserve retrieval ranking order — no set()
    sources = []
    for c in chunks_found:
        if c["source"] not in sources:
            sources.append(c["source"])

    # Return raw chunks to agent — no intermediate generate_answer()
    # ReAct synthesis LLM reasons over raw evidence directly
    sources_str  = ", ".join(sources)
    evidence_txt = " | ".join([c["text"][:200] for c in chunks_found[:3]])

    return (
        f"Retrieved {len(chunks_found)} chunks from: {sources_str}\n"
        f"Evidence: {evidence_txt}",
        sources,
        chunks_found
    )


# ── Tool 2: Summarise long text ──────────────────────────
def summarise_text_tool(text):
    """
    LangChain Tool 2: Summarises long retrieved text to stay
    within token limits during multi-step synthesis.
    Called automatically when a finding exceeds 600 characters.
    """
    prompt = (
        "Summarise this research text in exactly 3 sentences. "
        "Keep all key technical details, numbers, and exact paper filenames. "
        "NEVER add author names like Lewis et al. unless they appear in the text. "
        "Do not generalise — be specific:\n\n"
        f"{text[:1000]}\n\nSUMMARY:"
    )
    return langchain_invoke(_llm, prompt)


# ── Tool 3: Format citations ──────────────────────────────
def format_citations_tool(sources):
    """
    LangChain Tool 3: Maps source filenames to proper APA citations.
    Uses keyword matching on filename (e.g. 'lewis' → Lewis et al.).
    Covers all papers in the corpus (base 25 + any uploaded PDFs).
    Falls back to 'Source: filename' for uploaded/unknown papers.
    """
    citation_map = {
        "lewis"    : "Lewis et al. (2020). Retrieval-Augmented Generation. NeurIPS 33.",
        "yao"      : "Yao et al. (2023). ReAct: Synergizing Reasoning and Acting. ICLR.",
        "reimers"  : "Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019.",
        "es"       : "Es et al. (2024). RAGAS: Automated Evaluation of RAG. EACL 2024.",
        "wang"     : "Wang et al. (2025). Survey on LLM-Based Autonomous Agents. FCS.",
        "aytar"    : "Aytar et al. (2024). RAG Framework for Academic Literature. arXiv.",
        "singh"    : "Singh et al. (2025). Agentic Retrieval-Augmented Generation Survey.",
        "gao"      : "Gao et al. (2024). RAG for Large Language Models: A Survey.",
        "zhao"     : "Zhao et al. (2024). Retrieval-Augmented Generation and Beyond.",
        "devlin"   : "Devlin et al. (2019). BERT: Pre-training Deep Bidirectional Transformers. NAACL.",
        "wei"      : "Wei et al. (2022). Chain-of-Thought Prompting. NeurIPS 2022.",
        "johnson"  : "Johnson et al. (2021). Billion-Scale Similarity Search with FAISS. IEEE.",
        "karpukhin": "Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP.",
        "shinn"    : "Shinn et al. (2024). Reflexion: Language Agents with Verbal Reinforcement. NeurIPS.",
        "ji"       : "Ji et al. (2023). Survey of Hallucination in Natural Language Generation. ACM.",
        "brown"    : "Brown et al. (2020). Language Models are Few-Shot Learners (GPT-3). NeurIPS.",
        "touvron"  : "Touvron et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Models. Meta AI.",
        "yu"       : "Yu et al. (2024). Evaluation of Retrieval-Augmented Generation: A Survey.",
        "raffel"   : "Raffel et al. (2020). Exploring Limits of Transfer Learning with T5. JMLR.",
        "schick"   : "Schick et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools.",
    }
    citations = []
    for source in sources:
        matched = False
        for key, citation in citation_map.items():
            if key in source.lower():
                citations.append(citation)
                matched = True
                break
        if not matched:
            clean = source.replace('.pdf', '').replace('_', ' ')
            citations.append(f"Source: {clean}")
    # Preserve citation order — GPT suggestion
    ordered = []
    for c in citations:
        if c not in ordered:
            ordered.append(c)
    return ordered


# ── Synthesis prompt builders ─────────────────────────────
def _build_gap_synthesis(question, findings_block):
    """
    Specialized synthesis prompt for Research Gap Finder mode.
    Forces the LLM to output structured paper/limitation/future-work
    format as suggested in the ChatGPT improvement feedback.
    """
    return f"""You are an expert research analyst.

CITATION RULE: Only cite papers whose EXACT filename appears in the findings.
NEVER add author names like "Lewis et al." not present in the findings text.

CITATION RULE: Only cite papers whose EXACT filename appears in findings.
NEVER add "Lewis et al." unless verbatim in findings.

TASK: Extract specific research gaps from the paper findings below.
Only include gaps EXPLICITLY mentioned in the papers — do not invent.

QUESTION: {question}

FINDINGS FROM PAPERS:
{findings_block}

FORMAT YOUR ANSWER EXACTLY LIKE THIS — no deviations:

Research Gap 1
--------------
Paper: [exact paper name from findings]
Limitation: [specific limitation stated in that paper, in 1-2 sentences]
Future Work: [what that paper says researchers should do next]

Research Gap 2
--------------
Paper: [exact paper name from findings]
Limitation: [specific limitation stated in that paper, in 1-2 sentences]
Future Work: [what that paper says researchers should do next]

Research Gap 3
--------------
Paper: [exact paper name from findings]
Limitation: [specific limitation stated in that paper, in 1-2 sentences]
Future Work: [what that paper says researchers should do next]

Overall Research Direction
--------------------------
[2 complete sentences summarising the main open problems across all papers]

STRICT RULES:
- Use ONLY information from the findings above
- Name the EXACT paper for each gap
- Do NOT merge different papers' gaps into one point
- Complete every sentence fully — never cut off
- If fewer than 3 gaps are found, write fewer — do not invent

ANSWER:"""


def _build_methodology_synthesis(question, findings_block):
    """Methodology Finder — single paper handled correctly."""
    import re as _re
    paper_names = _re.findall(r"Paper:\s*(.+?)\n", findings_block)
    unique_papers = []
    for p in paper_names:
        p = p.strip()
        if p and p not in unique_papers:
            unique_papers.append(p)
    paper_count = len(unique_papers)
    single_note = "Only one relevant paper was retrieved. Cross-paper methodology comparison is not applicable.\n\n" if paper_count <= 1 else ""
    comp_section = "Comparison\n----------\nOnly one relevant paper was retrieved. Cross-paper methodology comparison is not applicable." if paper_count <= 1 else "Comparison\n----------\n[2 sentences comparing approaches across papers]"

    return f"""You are an expert research analyst.
TASK: Extract methodology details from paper findings.
QUESTION: {question}
FINDINGS: {findings_block}
{single_note}FORMAT:
Methodology Overview
--------------------
[1 sentence on what methodology area this covers]
Paper-by-Paper Breakdown
------------------------
Paper: [exact filename]
Method: [specific method or framework]
Dataset: [dataset used or "not specified"]
Key Result: [main finding with numbers]
Why Chosen: [reason authors chose this approach]
{comp_section}
Overall Conclusion
------------------
[2 sentences on what evidence shows]
RULES: Never write "Paper: Not specified". Never invent papers. Use exact filenames.
ANSWER:"""


def _extract_comparison_sides(question):
    """
    FIX (concept-based comparison): identify the two concepts/approaches
    the user actually asked to compare — e.g. "simple RAG" vs "agentic
    RAG" in "Compare simple RAG with agentic RAG for multi-hop
    questions" — so the synthesis prompt compares THOSE, instead of
    silently substituting whichever two paper filenames happened to be
    retrieved (the bug that turned a Simple-vs-Agentic-RAG question into
    a zhao2024_rag_aigc.pdf-vs-gao2024_rag_survey.pdf comparison).
    Returns (side_a, side_b) or (None, None) if no clear pattern is found.
    """
    q = question.strip()
    patterns = [
        r'compare\s+(.+?)\s+(?:with|to|and|versus|vs\.?)\s+(.+?)(?:\s+for\s|\s+in\s|\s+on\s|\?|$)',
        r'difference between\s+(.+?)\s+and\s+(.+?)(?:\s+for\s|\s+in\s|\s+on\s|\?|$)',
        r'how does\s+(.+?)\s+(?:improve|compare|differ)\s+over\s+(.+?)(?:\s+for\s|\s+in\s|\s+on\s|\?|$)',
        r'(.+?)\s+vs\.?\s+(.+?)(?:\s+for\s|\s+in\s|\s+on\s|\?|$)',
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            side_a = m.group(1).strip(" ,.")
            side_b = m.group(2).strip(" ,.")
            if side_a and side_b and len(side_a) < 60 and len(side_b) < 60:
                return side_a, side_b
    return None, None


def _build_comparison_synthesis(question, findings_block, side_a=None, side_b=None):
    """Compare Papers — Python paper count, single paper → summary."""
    import re as _re

    # FIX (concept-based comparison): when the question names two clear
    # approaches/concepts to compare, use a prompt that compares THOSE —
    # the retrieved papers are supporting evidence only, never the thing
    # being compared. Falls back to the old paper-vs-paper prompt only
    # when no clear concept pair could be parsed from the question (e.g.
    # an explicit "compare papers on X" request).
    if side_a and side_b:
        return f"""You are an expert research analyst.

The user asked to compare two APPROACHES/CONCEPTS — not two papers:
  Side A: {side_a}
  Side B: {side_b}

QUESTION: {question}

RETRIEVED EVIDENCE (supporting material only — these papers are NOT the
two things being compared; use them only as evidence FOR each side):
{findings_block}

STRICT RULES:
- Compare "{side_a}" vs "{side_b}" directly. Do NOT replace this with a
  comparison between the retrieved paper filenames.
- For EACH side, describe: mechanism, strengths, limitations, and
  suitability for the task named in the question.
- Cite the exact paper filename from the evidence above wherever you
  use a specific fact — but the comparison itself is between the two
  approaches, not between papers.
- If evidence for one side is thin or missing, say so explicitly
  (e.g. "Limited evidence retrieved for {side_b} in this corpus")
  instead of inventing a comparison or silently comparing something else.
- Never fabricate a number, percentage, or score not stated verbatim
  in the evidence above.

FORMAT EXACTLY:

{side_a}
{'-' * len(side_a)}
Mechanism: [...]
Strengths: [...]
Limitations: [...]
Suitability for this task: [...]

{side_b}
{'-' * len(side_b)}
Mechanism: [...]
Strengths: [...]
Limitations: [...]
Suitability for this task: [...]

Head-to-Head
------------
[2-3 sentences directly comparing {side_a} and {side_b} on what the question asks]

Overall Conclusion
------------------
[1-2 sentences: which is better suited, per the evidence — or say the evidence doesn't decide it]
ANSWER:"""

    paper_names = _re.findall(r"Paper:\s*(.+?)\n", findings_block)
    unique_papers = []
    for p in paper_names:
        p = p.strip()
        if p and p not in unique_papers:
            unique_papers.append(p)
    paper_count = len(unique_papers)

    if paper_count <= 1:
        pname = unique_papers[0] if unique_papers else "the retrieved paper"
        return f"""You are an expert research analyst.
ONLY ONE paper found: {pname}. Write summary ONLY — no research gaps.
FINDINGS: {findings_block}
Write EXACTLY:
Only one relevant paper was retrieved.
Paper comparison requires at least two papers.
Summary of retrieved paper:
PAPER: {pname}
MAIN CLAIM: [what the paper argues — 1-2 sentences with specifics]
KEY RESULT: [most important finding with exact numbers]
STRENGTH: [what this paper does methodologically well]
LIMITATION: [one limitation explicitly stated in the paper]
ANSWER:"""
    else:
        return f"""You are an expert research analyst.
Compare {paper_count} papers on: {question}
Papers: {", ".join(unique_papers)}
FINDINGS: {findings_block}
Write EXACTLY:
Key Similarities
----------------
- [Agreement — name BOTH papers with exact filenames]
Key Differences
---------------
- [Difference — Paper A vs Paper B — include a number ONLY if it
  appears verbatim in the evidence above; if no comparable numeric
  result exists for both papers, describe the difference in words
  instead — do NOT invent a score, percentage, or decimal number]
Summary Table
-------------
Paper | Key Method | Key Finding
[exact filename] | [method] | [finding — number only if stated in evidence, else describe in words]
Overall Conclusion
------------------
[2 sentences on what papers collectively show]
RULES: Exact filenames. No invented citations. NEVER fabricate a
numeric score, percentage, or decimal metric that does not appear
verbatim in the evidence above — a qualitative comparison is
better than a made-up number.
ANSWER:"""


def _build_general_synthesis(question, findings_block):
    """
    General synthesis for Deep Research mode.
    Reasons over RAW paper evidence, not LLM summaries.
    """
    return f"""You are an expert research analyst.

QUESTION: {question}

RETRIEVED EVIDENCE FROM PAPERS:
{findings_block}

CRITICAL RULES — FOLLOW EXACTLY:
1. CITATIONS: Only use EXACT filenames from the evidence above.
   NEVER write author names like "Lewis et al." unless verbatim in evidence.
2. MECHANISM: If evidence mentions "labor displacement", "opportunity cost",
   "mechanization", or "inequality" — state these AS the mechanism.
   NEVER say "mechanism not established" if these terms appear.
3. NUMBERS: Include specific numbers, %, or coefficients ONLY if they
   appear verbatim in the evidence above. If no such number exists for
   a point, describe it qualitatively instead — NEVER invent, estimate,
   or guess a number, percentage, or score that is not explicitly stated.
4. LENGTH: Maximum 4 sentences per finding. Total answer under 250 words.
5. ROLE-AWARE READING: Some findings above are tagged "(addresses:
   mechanism)" or "(addresses: measurement)" — this means that finding
   was retrieved specifically to answer ONE part of a multi-part
   question. Only use a tagged finding to answer the part of the
   question it is tagged for. Do NOT use "measurement"-tagged evidence
   to explain how something works, and do NOT use "mechanism"-tagged
   evidence to explain how something is evaluated. If the question has
   multiple parts, address each part separately and explicitly before
   connecting them in SYNTHESIS.

YOUR ANSWER STRUCTURE:

1. DIRECT ANSWER:
[1-2 sentences with core answer; include a number only if the evidence states one]

2. EVIDENCE FROM PAPERS:
--- Finding 1 ---
Paper: [exact filename from evidence ONLY]
Finding: [specific finding — numbers only if stated verbatim in evidence]

--- Finding 2 ---
Paper: [exact filename from evidence ONLY]
Finding: [specific finding — numbers only if stated verbatim in evidence]

3. SYNTHESIS:
[2 sentences connecting findings]

4. CONCLUSION:
[1 sentence; include a number only if the evidence actually states one]

RULES:
- Every claim must come from the evidence above
- Quote or closely paraphrase actual paper text
- Do not add information not in the evidence
- Do NOT mention any specific project name
- Maximum 300 words
- Be specific with paper names and numbers
"""


def react_agent(question, model, index, chunks,
                client, max_searches=3,
                uploaded_sources=None,
                search_scope="all",
                force_mode=None):
    """
    LangChain-powered ReAct agent implementing the
    Thought → Action → Observation loop (Yao et al. 2023).

    Improvements over standard single-step RAG:
    1. Plans multiple targeted search queries
    2. Detects question type → uses specialized synthesis prompt
    3. Synthesises findings from multiple papers into structured output
    4. Formats citations using LangChain tool pattern

    This directly addresses the gap identified in Aytar et al. (2024)
    who noted their RAG system lacked agentic multi-step reasoning.
    """
    global _model, _index, _chunks, _client, _llm
    global _uploaded_sources
    global _search_scope
    _model            = model
    _index            = index
    _chunks           = chunks
    _client           = client
    _llm              = get_langchain_llm()
    # uploaded_sources passed from app.py via react_agent
    _uploaded_sources = uploaded_sources
    _search_scope     = search_scope

    print(f"\n{'='*60}")
    print(f"LANGCHAIN REACT AGENT")
    print(f"Question: {question}")
    print(f"{'='*60}")

    all_sources  = []
    all_findings = []
    steps_trace  = []

    # ── Extract raw question FIRST then detect type ──────
    # WHY extract first?
    # app.py wraps question with gap_prompt/compare_prompt templates
    # These templates contain words like "limitation", "missing"
    # which falsely trigger is_gap even for Compare Papers mode.
    # Solution: strip wrapper FIRST, detect type on raw question only.

    # Step A: Strip prompt wrappers
    raw_q_for_detect = question
    q_low_full = question.lower()
    if "research gaps about:" in q_low_full:
        raw_q_for_detect = question.split("research gaps about:")[-1].strip().split("\n")[0].strip()
    elif "asking about:" in q_low_full:
        raw_q_for_detect = question.split("asking about:")[-1].strip().split("\n")[0].strip()
    elif "wants to compare" in q_low_full:
        idx = q_low_full.find("wants to compare")
        raw_q_for_detect = question[idx+len("wants to compare"):].strip().split("\n")[0].strip()
        if raw_q_for_detect.lower().startswith("papers on:"):
            raw_q_for_detect = raw_q_for_detect[len("papers on:"):].strip()
    elif "compare papers on:" in q_low_full:
        raw_q_for_detect = question.split("compare papers on:")[-1].strip().split("\n")[0].strip()
    elif "about:" in q_low_full and len(question) > 100:
        raw_q_for_detect = question.split("about:")[-1].strip().split("\n")[0].strip()

    # Step B: Detect type from RAW question only (not wrapped prompt)
    # force_mode from app.py overrides keyword detection
    # This prevents "framework" triggering methodology in Deep Research mode
    if force_mode == "deep":
        is_gap = False
        is_methodology = False
        is_comparison = False
        q_lower = raw_q_for_detect.lower()
    elif force_mode == "gap":
        is_gap = True
        is_methodology = False
        is_comparison = False
        q_lower = raw_q_for_detect.lower()
    elif force_mode == "methodology":
        is_gap = False
        is_methodology = True
        is_comparison = False
        q_lower = raw_q_for_detect.lower()
    elif force_mode == "comparison":
        is_gap = False
        is_methodology = False
        is_comparison = True
        q_lower = raw_q_for_detect.lower()
    else:
        q_lower = raw_q_for_detect.lower()
        is_gap = any(w in q_lower for w in [
            "gap", "limitation", "future work", "drawback",
            "challenge", "weakness", "missing", "lacks",
            "problem", "issue", "shortcoming"
        ])
        # FIX (mode detection was over-triggering): "method", "approach",
        # "framework", "process" are generic nouns that show up in normal
        # explanatory/comparison questions too (e.g. "How does ReAct
        # framework improve over simple RAG?" was wrongly routed into
        # Methodology Finder mode, which then pulled in Reflexion,
        # instruction-tuning, and other unrelated papers). Comparison is
        # now detected first with stronger phrase-level signals, and
        # Methodology mode requires explicit methodology-seeking intent
        # rather than any mention of the word "method"/"framework".
        is_comparison = any(w in q_lower for w in [
            "compare", "comparison", "difference between", " vs ", " vs.",
            "versus", "contrast", "better than", "improve over",
            "improves over", "compared to", "compared with",
            "which is better", "how do they differ", "outperform"
        ])
        is_methodology = (not is_comparison) and any(w in q_lower for w in [
            "methodology", "what method", "which method",
            "methods used", "dataset used", "experimental setup",
            "model architecture", "how was", "how did",
            "evaluation protocol", "training procedure",
            "implementation detail"
        ])

    # ── Step 1: Build search queries in Python (no LLM) ────────
    # WHY bypass LLM planning entirely?
    # LLM generated wrong queries: "seed germination", "site:sciencedirect.com"
    # Python extraction from question words is 100% reliable.

    stop = {
        "what","is","the","a","an","of","in","to","how","does",
        "do","did","are","was","were","and","or","for","with","that",
        "this","these","those","can","could","would","should",
        "has","have","had","be","been","being","its","it","at",
        "by","from","as","on","their","they","we","our","my",
        "me","which","when","where","who","why","will","just",
        "also","used","use","using","between","summarise","summary",
        "compare","paper","papers","tell","give","show","find",
        "make","please","your","their","there","about","help",
        "you","expert","academic","analyst","researcher","study",
        "research","question","answer","provide","explain","describe"
    }

    # Strip prompt wrappers added by app.py gap_prompt/method_prompt/compare_prompt
    raw_question = question
    q_low = question.lower()
    if "research gaps about:" in q_low:
        idx = q_low.find("research gaps about:")
        raw_question = question[idx+len("research gaps about:"):].strip().split("\n")[0].strip()
    elif "asking about:" in q_low:
        idx = q_low.find("asking about:")
        raw_question = question[idx+len("asking about:"):].strip().split("\n")[0].strip()
    elif "compare papers on:" in q_low:
        idx = q_low.find("compare papers on:")
        raw_question = question[idx+len("compare papers on:"):].strip().split("\n")[0].strip()
    elif "wants to compare" in q_low:
        idx = q_low.find("wants to compare")
        raw_question = question[idx+len("wants to compare"):].strip().split("\n")[0].strip()
        if raw_question.startswith("papers on:"):
            raw_question = raw_question[len("papers on:"):].strip()
    elif "about:" in q_low and len(question) > 100:
        idx = q_low.find("about:")
        raw_question = question[idx+len("about:"):].strip().split("\n")[0].strip()

    print(f"Extracted question: {raw_question[:80]}")

    words = [w for w in re.sub(r'[^a-zA-Z0-9 ]', ' ', raw_question).split()
             if w.lower() not in stop and len(w) > 2]

    # FIX (entity-preserving queries): "topic" used to be just the first
    # 3 non-stopword tokens in sentence order. For "What methodology did
    # Reimers and Gurevych use in SBERT?" that produced "methodology did
    # Reimers" — silently dropping "Gurevych" and "SBERT" because they
    # appear later in the sentence. Every generated search then missed
    # the one paper that actually answers the question. Entities (proper
    # nouns / short acronyms) are extracted first and always kept in the
    # topic regardless of their position in the sentence.
    def _is_entity(w):
        return (w.isupper() and len(w) <= 6) or w[0].isupper()

    seen_e, entities = set(), []
    for w in words:
        if _is_entity(w) and w not in seen_e:
            seen_e.add(w); entities.append(w)
    non_entity_words = [w for w in words if w not in entities]

    topic_terms = (entities + non_entity_words)[:5]
    topic = ' '.join(topic_terms) if topic_terms else raw_question[:30]

    search_queries_roles = None  # set explicitly only in the multi-part branch below

    if is_gap:
        search_queries = [
            topic + ' limitation drawback',
            topic + ' exclusion restriction robustness',
            topic + ' future research direction',
        ]
    elif is_methodology:
        search_queries = [
            topic + ' method framework',
            topic + ' dataset instrument',
            topic + ' result coefficient finding',
        ]
    elif is_comparison:
        mid = max(1, len(words)//2)
        search_queries = [
            ' '.join(words[:min(4,mid+1)]) if words else raw_question[:40],
            ' '.join(words[mid:mid+4]) if len(words)>mid else ' '.join(words[-3:]),
            topic + ' comparison performance',
        ]
    else:
        robustness_terms = ["robust","placebo","parallel","check","test","valid"]
        # FIX (query decomposition, role-aware): multi-part questions like
        # "How does RAG reduce hallucination and how is this measured?"
        # previously produced near-duplicate searches because both were
        # sliding windows over the same word list — the second clause
        # never got its own targeted search, and evidence from one clause
        # could bleed into conclusions about the other (e.g. concluding
        # that "evaluation reduces hallucination"). Split on "and" into
        # real sub-questions, keep each clause's own entities, boost thin
        # measurement/evaluation clauses with concrete eval terms, and
        # tag each search with a role so its evidence stays scoped to the
        # part of the question it actually answers.
        and_parts = [p.strip() for p in re.split(r'\band\b', raw_question, flags=re.IGNORECASE) if p.strip()]
        if len(and_parts) >= 2:
            search_queries, search_queries_roles = [], []
            for part in and_parts[:max_searches]:
                p_words = [w for w in re.sub(r'[^a-zA-Z0-9 ]', ' ', part).split()
                          if w.lower() not in stop and len(w) > 2]
                p_entities = [w for w in p_words if _is_entity(w)]
                p_rest = [w for w in p_words if w not in p_entities]
                q_terms = (p_entities + p_rest)[:5]
                q = ' '.join(q_terms) if q_terms else part[:40]
                is_measurement = any(t in part.lower() for t in
                                     ["measur", "evaluat", "metric", "assess", "faithful"])
                if is_measurement:
                    q += " evaluation metrics"
                search_queries.append(q)
                search_queries_roles.append("measurement" if is_measurement else "mechanism")
        elif any(t in q_lower for t in robustness_terms):
            search_queries = [
                topic,
                'placebo test exclusion restriction',
                'parallel trend pre trend robustness',
            ]
        elif len(words) >= 5:
            search_queries = [' '.join(words[:5]), ' '.join(words[2:7]), ' '.join(words[-4:])]
        else:
            search_queries = [topic, ' '.join(words), raw_question[:40]]

    if search_queries_roles is None:
        search_queries_roles = ["general"] * len(search_queries)

    # Deduplicate and limit — keep roles aligned with their queries
    seen_q, clean_q, clean_roles = set(), [], []
    for q, role in zip(search_queries, search_queries_roles):
        q = q.strip()[:60]
        if q and q not in seen_q:
            seen_q.add(q); clean_q.append(q); clean_roles.append(role)
    search_queries = clean_q[:max_searches] or [topic]
    query_roles    = clean_roles[:max_searches] or ["general"]

    print(f"\nPlanned {len(search_queries)} searches (Python-built):")
    for i, q in enumerate(search_queries):
        print(f"  {i+1}. {q}")

    # ── Step 2: Execute searches (ReAct Action steps) ─────
    print("\nStep 2: Executing searches...")
    for i, query in enumerate(search_queries):
        print(f"\n  Search {i+1}/{len(search_queries)}: {query[:60]}...")

        # THOUGHT: what we are looking for
        thought = f"Searching for: {query}"

        # ACTION: call the RAG search tool
        # Returns (answer, sources, raw_chunks)
        sr = search_papers_tool(query)
        finding = str(sr[0])
        sources = sr[1]
        raw_chunks = sr[2] if len(sr) > 2 else []

        # Build raw evidence from actual paper text
        # Build raw evidence — sort by score, take top 3
        role = query_roles[i] if i < len(query_roles) else "general"
        raw_candidates = []
        for ch in raw_chunks:
            if isinstance(ch, dict) and 'text' in ch:
                score = float(ch.get('score', 0.0))
                ev_words = ch['text'].split()[:300]
                raw_candidates.append({
                    "source": ch.get('source', 'unknown'),
                    "text"  : ' '.join(ev_words),
                    "score" : score,
                    "role"  : role
                })

        # Sort by score descending
        raw_candidates = sorted(
            raw_candidates, key=lambda x: x['score'], reverse=True
        )

        # FIXED: relevance filtering is now RELATIVE to the best score
        # in this specific search, not a fixed absolute number.
        # WHY: SBERT cosine scores for genuinely relevant chunks can
        # sit anywhere in the 0.15-0.5 range depending on the corpus —
        # a fixed cutoff like 0.30 was discarding good, relevant
        # evidence and causing false "insufficient evidence" answers.
        # ABS_FLOOR only catches truly near-zero/irrelevant matches;
        # REL_CUTOFF keeps anything reasonably close to the top hit.
        ABS_FLOOR  = 0.15   # near-zero similarity = truly unrelated
        REL_CUTOFF = 0.5    # keep chunks scoring >= 50% of the best hit

        top_score = raw_candidates[0]['score'] if raw_candidates else 0.0
        raw_evidence = [
            ev for ev in raw_candidates
            if ev['score'] >= ABS_FLOOR and ev['score'] >= top_score * REL_CUTOFF
        ]

        # Source diversity — only enforce the "max 2 per paper" cap
        # when evidence genuinely spans MULTIPLE papers. If only one
        # paper is relevant (common for Methodology Finder / narrow
        # topics), keep up to 3 chunks from it instead of discarding
        # good evidence for the sake of diversity.
        distinct_sources = len(set(ev['source'] for ev in raw_evidence))
        if distinct_sources > 1:
            seen_src = {}
            diverse_evidence = []
            for ev in raw_evidence:
                cnt = seen_src.get(ev['source'], 0)
                if cnt < 2:
                    diverse_evidence.append(ev)
                    seen_src[ev['source']] = cnt + 1
                if len(diverse_evidence) >= 3:
                    break
            raw_evidence = diverse_evidence
        else:
            raw_evidence = raw_evidence[:3]

        # Summarise LLM answer if too long (keep raw evidence intact)
        if len(finding) > 600:
            print(f"  Summarising LLM answer ({len(finding)} chars)...")
            finding = str(summarise_text_tool(finding))

        # OBSERVATION: what we found
        print(f"  Sources found: {sources}")
        print(f"  Finding preview: {finding[:120]}...")

        all_sources.extend(sources)
        all_findings.append({
            "query"       : query,
            "finding"     : finding,
            "sources"     : sources,
            "raw_evidence": raw_evidence
        })
        steps_trace.append({
            "step"       : i + 1,
            "thought"    : thought,
            "action"     : f"SEARCH[{query}]",
            "observation": finding[:400]
        })

    # ── NEW: Insufficient evidence guard ──────────────────
    # If NO chunk across all searches passed the relevance
    # threshold, don't let the LLM invent an answer — return
    # a clear "insufficient evidence" response instead.
    total_evidence = sum(len(f.get('raw_evidence', [])) for f in all_findings)
    if total_evidence == 0:
        print("\n⚠️ No sufficiently relevant evidence found — skipping synthesis")
        return {
            "question"       : question,
            "answer"         : (
                "Insufficient evidence in the indexed papers to answer "
                "this question confidently. Try rephrasing, or upload a "
                "paper directly relevant to this topic."
            ),
            "citations"      : [],
            "sources"        : [],
            "evidence"       : [],
            "steps_taken"    : len(steps_trace),
            "reasoning_trace": steps_trace
        }

    # ── Step 3: Synthesise all findings ───────────────────
    print("\nStep 3: Synthesising with specialised prompt...")

    # Per-snippet 150-word limit — preserves balance across papers
    def limit_snippet(text, max_words=150):
        wds = text.split()
        return ' '.join(wds[:max_words]) + ('...' if len(wds)>max_words else '')

    # FIX (evidence quality over quantity): previously every search kept
    # up to 3 chunks and the prompt pulled 2 per finding across up to 3
    # searches — 5-8 sources reaching synthesis, often duplicated across
    # searches (same paper, same passage) or garbled (OCR/merged-word
    # text). Judge scores showed this hurt quality: Agentic RAG used
    # more sources on every question (5/5) but scored lower than Simple
    # RAG (1-2 sources) on 4/5. Pool evidence across ALL searches first,
    # drop duplicates and garbled chunks, then hard-cap to the strongest
    # few pieces of evidence — closer to how Simple RAG stays focused.
    def _is_garbled(text):
        wds = text.split()
        if not wds:
            return True
        long_words = [w for w in wds if len(w) > 20]  # merged/OCR tokens
        return (len(long_words) / len(wds)) > 0.15

    pooled, seen_keys = [], set()
    for f in all_findings:
        for ev in f.get('raw_evidence', []):
            key = (ev['source'], ev['text'][:60].lower())
            if key in seen_keys or _is_garbled(ev['text']):
                continue
            seen_keys.add(key)
            pooled.append(ev)
    pooled.sort(key=lambda x: x['score'], reverse=True)

    # FIX (evidence sufficiency gate): relevance-score filtering alone let
    # through evidence that was topically adjacent (RAG/LLM-related) but
    # never actually contained the specific entity the question named —
    # e.g. asking about Reimers/Gurevych/SBERT but synthesising an answer
    # from Toolformer/agent-survey papers because the query planner had
    # dropped those entities. A fluent answer about the wrong paper is
    # worse than admitting the evidence is insufficient. If the question
    # names specific entities, require at least one of them to appear
    # somewhere in the pooled evidence; if not, retry once with a query
    # built purely from those entities before giving up.
    if entities:
        combined_text = " ".join(f"{ev['source']} {ev['text']}" for ev in pooled).lower()
        entity_hit = any(e.lower() in combined_text for e in entities)
        if not entity_hit:
            print(f"  ⚠️ Named entities {entities} not found in retrieved "
                  f"evidence — retrying with an entity-focused search")
            retry_sr = search_papers_tool(' '.join(entities))
            retry_chunks = retry_sr[2] if len(retry_sr) > 2 else []
            retry_candidates = []
            for ch in retry_chunks:
                if isinstance(ch, dict) and 'text' in ch:
                    retry_candidates.append({
                        "source": ch.get('source', 'unknown'),
                        "text"  : ' '.join(ch['text'].split()[:300]),
                        "score" : float(ch.get('score', 0.0)),
                        "role"  : "retry"
                    })
            retry_candidates.sort(key=lambda x: x['score'], reverse=True)
            retry_hit = any(
                e.lower() in (ev['source'] + ' ' + ev['text']).lower()
                for ev in retry_candidates for e in entities
            )
            if retry_hit:
                for ev in retry_candidates:
                    if not any(e.lower() in (ev['source']+' '+ev['text']).lower() for e in entities):
                        continue
                    key = (ev['source'], ev['text'][:60].lower())
                    if key in seen_keys or _is_garbled(ev['text']):
                        continue
                    seen_keys.add(key)
                    pooled.append(ev)
                pooled.sort(key=lambda x: x['score'], reverse=True)
                print(f"  ✅ Retry found matching evidence for {entities}")
            else:
                print(f"  ⚠️ Retry search still found no evidence for "
                      f"{entities} — returning insufficient-evidence response")
                return {
                    "question"       : question,
                    "answer"         : (
                        f"Insufficient relevant evidence was retrieved for "
                        f"{', '.join(entities)} in the indexed papers. "
                        f"They may not be covered in enough depth in this "
                        f"corpus — try rephrasing, or upload a paper "
                        f"directly relevant to this subject."
                    ),
                    "citations"      : [],
                    "sources"        : [],
                    "evidence"       : [],
                    "steps_taken"    : len(steps_trace),
                    "reasoning_trace": steps_trace
                }

    MAX_FINAL_EVIDENCE = 4
    top_evidence = pooled[:MAX_FINAL_EVIDENCE]

    limited = ""
    if top_evidence:
        # FIX (role-aware evidence): each item keeps the role of the
        # search that found it (e.g. "mechanism" vs "measurement" for
        # multi-part questions) so the synthesis prompt can be told to
        # only use each piece of evidence for the part of the question
        # it actually addresses, instead of letting evidence for "how is
        # this measured" get folded into the explanation of "how does it
        # work".
        for i, ev in enumerate(top_evidence):
            role = ev.get('role', 'general')
            role_tag = f" (addresses: {role})" if role not in ("general", "retry") else ""
            limited += f"\n--- Finding {i+1}{role_tag} ---\n"
            # FIX: removed the raw cosine-similarity "Score" field
            # from what the LLM sees — it was an internal retrieval
            # ranking number, not a result from the paper, but the
            # LLM was hallucinating it into answers as if it were a
            # reported metric (e.g. "Score: 0.568 vs 0.367").
            limited += f"Paper: {ev['source']}\nEvidence: {limit_snippet(ev['text'], 100)}\n---\n"
    else:
        # Fallback: no clean raw_evidence survived filtering — use the
        # LLM-summarised findings instead of leaving the prompt empty.
        for i, f in enumerate(all_findings):
            limited += f"\n--- Finding {i+1} ---\nQuery: {f['query']}\nContent: {limit_snippet(f['finding'], 100)}\n"
    findings_block = limited

    # Choose synthesis prompt based on question type
    if is_gap:
        print("  → Using Research Gap synthesis prompt")
        synthesis_prompt = _build_gap_synthesis(question, findings_block)
    elif is_methodology:
        print("  → Using Methodology synthesis prompt")
        synthesis_prompt = _build_methodology_synthesis(question, findings_block)
    elif is_comparison:
        # FIX (concept-based comparison): compare the approaches/concepts
        # named in the question (e.g. "simple RAG" vs "agentic RAG"), not
        # whichever two paper filenames happened to be retrieved.
        side_a, side_b = _extract_comparison_sides(raw_question)
        if side_a and side_b:
            print(f"  → Using Comparison synthesis prompt "
                  f"(concept-based: '{side_a}' vs '{side_b}')")
        else:
            print("  → Using Comparison synthesis prompt (paper-based fallback)")
        synthesis_prompt = _build_comparison_synthesis(question, findings_block, side_a, side_b)
    else:
        print("  → Using General synthesis prompt")
        synthesis_prompt = _build_general_synthesis(question, findings_block)

    final_answer = langchain_invoke(_llm, synthesis_prompt)

    # ── Step 4: Format citations ───────────────────────────
    # FIX: cite from the capped/deduped top_evidence pool (the papers
    # actually used in synthesis) rather than every source touched
    # across all searches — otherwise citations balloon to 5-8 papers
    # even when only 2-4 genuinely informed the final answer.
    unique_sources = []
    source_pool = [ev['source'] for ev in top_evidence] if top_evidence else all_sources
    for s in source_pool:
        if s not in unique_sources:
            unique_sources.append(s)
    citations = format_citations_tool(unique_sources)

    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(final_answer)
    print(f"\nCITATIONS ({len(citations)}):")
    for c in citations:
        print(f"  → {c}")
    print(f"Steps completed: {len(steps_trace)}")
    print(f"{'='*60}")

    return {
        "question"       : question,
        "answer"         : final_answer,
        "citations"      : citations,
        "sources"        : unique_sources,
        # FIX: expose the exact evidence used for synthesis (post pooling/
        # dedup/garbled-filter/cap) — not the raw per-search reasoning_trace
        # observations, which were only intermediate LLM-summarised findings
        # and didn't reflect what actually informed the final answer.
        # 'score' stays here for eval/debugging purposes only — it is never
        # passed into the answer-generation prompt.
        "evidence"       : [
            {"source": ev["source"], "text": ev["text"], "score": ev["score"]}
            for ev in top_evidence
        ],
        "steps_taken"    : len(steps_trace),
        "reasoning_trace": steps_trace
    }


# ── Main: Test Phase 4 ────────────────────────────────────
if __name__ == "__main__":
    from phase2_embed import load_embedding_model, load_index
    from phase3_rag import setup_llm

    print("Loading Phase 4 — LangChain ReAct Agent...")
    model         = load_embedding_model()
    index, chunks = load_index()
    client        = setup_llm()

    os.makedirs("outputs", exist_ok=True)

    test_questions = [
        "What limitations and future work do RAG papers identify?",
        "What methodology did Reimers and Gurevych use in SBERT?",
        "Compare simple RAG with agentic RAG for multi-hop questions",
        "How does RAG reduce hallucination and how is this measured?",
        "What is the ReAct framework and how does it relate to LangChain?",
    ]

    all_results = []
    for question in test_questions:
        result = react_agent(
            question, model, index, chunks, client
        )
        all_results.append({
            "question" : result["question"],
            "answer"   : result["answer"],
            "citations": result["citations"],
            "steps"    : result["steps_taken"]
        })

    with open("outputs/agent_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved to outputs/agent_results.json")
    print("\nPhase 4 Complete — LangChain ReAct Agent working!")
