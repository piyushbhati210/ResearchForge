# ============================================================
# PHASE 5 — Final Evaluation (Complete Version, bug-fixed)
# ResearchForge | Piyush Bhati | SIG 2025-26
# ============================================================
# THREE-PART EVALUATION:
# Part 1: Functional/Operational Evaluation (20 questions)
# Part 2: RAGAS Automated Scoring (faithfulness + relevancy)
# Part 3: Simple RAG vs Agentic RAG (LLM-as-Judge)
# ============================================================
import os, time, json, re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

os.makedirs("outputs", exist_ok=True)


# ── Test Questions ────────────────────────────────────────
def get_test_questions():
    return [
        "What is Retrieval Augmented Generation?",
        "How does RAG reduce hallucination in LLMs?",
        "What are RAG-Sequence and RAG-Token models?",
        "How does dense passage retrieval work?",
        "What is the difference between parametric and non-parametric memory in RAG?",
        "How does SBERT improve over standard BERT?",
        "What is cosine similarity used for in SBERT?",
        "How fast is SBERT compared to BERT for semantic search?",
        "What is the ReAct framework for language models?",
        "How does ReAct combine reasoning and acting?",
        "What is the Thought Action Observation loop in ReAct?",
        "What is the Planning Memory Action framework for LLM agents?",
        "How do LLM agents use memory modules?",
        "What are the three RAGAS evaluation metrics?",
        "How does RAGAS measure faithfulness?",
        "What is answer relevance in RAGAS?",
        "What is semantic chunking in RAG?",
        "How does chain of thought prompting work?",
        "What is the Reflexion framework for language agents?",
        "What is the difference between short and long term memory in agents?",
    ]


# ── Safe RAG with retry ───────────────────────────────────
def safe_rag(question, model, index, chunks, client,
             top_k=3, max_words=80):
    """
    Safe RAG call with retry on rate limits.
    Reduces top_k and context size on each retry.
    """
    from phase3_rag import rag_pipeline

    for attempt, (tk, mw) in enumerate([(top_k, max_words), (2, 60), (1, 40)]):
        try:
            result = rag_pipeline(
                question, model, index, chunks, client, top_k=tk
            )
            contexts = []
            for ch in result['chunks_used']:
                words = ch['text'].split()[:mw]
                contexts.append(' '.join(words))
            return result['answer'], contexts, result['sources']

        except Exception as e:
            err = str(e)
            if '413' in err or 'rate_limit' in err.lower():
                wait = 20 + attempt * 10
                print(f"  ⚠️ Rate limit attempt {attempt+1} — wait {wait}s, tk={max(1,tk-1)}...")
                time.sleep(wait)
            elif 'Connection' in err:
                print(f"  ⚠️ Connection error attempt {attempt+1} — wait 10s...")
                time.sleep(10)
            else:
                print(f"  ❌ Error: {err[:60]}")
                break

    return "Could not generate answer.", ["No context retrieved."], []


# ── PART 1: Functional Evaluation ────────────────────────
def run_functional_eval(questions, model, index, chunks, client):
    """
    Runs all 20 questions and records operational statistics.
    Reports: answer rate, avg length, avg contexts.
    NOTE: This is operational/completion rate — NOT accuracy.
    """
    print("\n" + "="*60)
    print("PART 1: FUNCTIONAL EVALUATION (20 Questions)")
    print("="*60)
    print("Measures operational answer rate — NOT factual accuracy\n")

    q_list, a_list, c_list, src_list = [], [], [], []
    answered = 0

    for i, q in enumerate(questions):
        print(f"[{i+1:02d}/20] {q[:55]}...")
        answer, contexts, sources = safe_rag(
            q, model, index, chunks, client
        )
        q_list.append(q)
        a_list.append(answer)
        c_list.append(contexts)
        src_list.append(sources)

        ok = "could not" not in answer.lower()
        if ok: answered += 1
        print(f"  {'✅' if ok else '⚠️'} {len(answer)} chars | "
              f"{len(contexts)} contexts")
        time.sleep(8)

    # Save functional results
    df = pd.DataFrame({
        "sl_no"         : range(1, len(q_list)+1),
        "question"      : q_list,
        "answer_preview": [a[:100]+"..." for a in a_list],
        "answer_length" : [len(a) for a in a_list],
        "num_contexts"  : [len(c) for c in c_list],
        "num_sources"   : [len(s) for s in src_list],
        "answered"      : ["Yes" if "could not" not in a.lower()
                           else "No" for a in a_list],
    })
    df.to_csv("outputs/functional_results.csv", index=False)

    print("\n" + "-"*60)
    print(f"FUNCTIONAL EVALUATION RESULTS — TABLE 1")
    print("-"*60)
    print(f"  Total Questions        : {len(q_list)}")
    print(f"  Operational Answers    : {answered}/{len(q_list)}")
    print(f"  Operational Rate       : {answered/len(q_list)*100:.1f}%")
    print(f"  Avg Answer Length      : {df['answer_length'].mean():.0f} chars")
    print(f"  Avg Contexts Retrieved : {df['num_contexts'].mean():.1f}")
    print(f"  NOTE: Rate = completion, NOT factual accuracy")
    print("-"*60)
    print(f"✅ Saved → outputs/functional_results.csv")

    return q_list, a_list, c_list


# ── PART 2: RAGAS Evaluation ──────────────────────────────
def run_ragas_eval(q_list, a_list, c_list):
    """
    RAGAS with faithfulness + answer_relevancy only.
    context_precision skipped — requires reference answers.
    Always saves functional results even if RAGAS fails.
    """
    print("\n" + "="*60)
    print("PART 2: RAGAS AUTOMATED EVALUATION")
    print("="*60)

    try:
        os.environ["OPENAI_API_KEY"] = "dummy-not-used"

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        print("Configuring RAGAS with Groq + SBERT...")

        groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=300
        )
        ragas_llm = LangchainLLMWrapper(groq_llm)

        # Try embedding wrappers in order
        ragas_emb = None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            hf = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            ragas_emb = LangchainEmbeddingsWrapper(hf)
            print("Using langchain-huggingface embeddings")
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from ragas.embeddings import LangchainEmbeddingsWrapper
                hf = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )
                ragas_emb = LangchainEmbeddingsWrapper(hf)
                print("Using langchain-community embeddings")
            except ImportError:
                from sentence_transformers import SentenceTransformer
                from ragas.embeddings import LangchainEmbeddingsWrapper

                class SBERTEmb:
                    def __init__(self):
                        self.m = SentenceTransformer(
                            'all-MiniLM-L6-v2', local_files_only=True
                        )
                    def embed_documents(self, texts):
                        return self.m.encode(
                            texts, normalize_embeddings=True
                        ).tolist()
                    def embed_query(self, text):
                        return self.m.encode(
                            [text], normalize_embeddings=True
                        )[0].tolist()

                ragas_emb = LangchainEmbeddingsWrapper(SBERTEmb())
                print("Using sentence-transformers embeddings")

        # Apply LLM and embeddings to metrics
        faithfulness.llm        = ragas_llm
        faithfulness.embeddings = ragas_emb
        answer_relevancy.llm        = ragas_llm
        answer_relevancy.embeddings = ragas_emb

        # context_precision is deliberately NOT attempted here:
        # it requires a 'reference' (ground-truth) column that
        # this dataset does not have, and previous runs showed it
        # fails the whole evaluate() call rather than failing
        # gracefully on its own. It is documented as a limitation
        # instead (see report Section 5.5).
        metrics = [faithfulness, answer_relevancy]

        dataset = Dataset.from_dict({
            "question": q_list,
            "answer"  : a_list,
            "contexts": c_list
        })

        print(f"Scoring {len(q_list)} samples...")
        print("This takes 5-10 minutes...\n")

        results = evaluate(dataset=dataset, metrics=metrics)
        df_ragas = results.to_pandas()

        print("\n" + "-"*60)
        print("RAGAS RESULTS — TABLE 2")
        print("-"*60)
        for col in ['faithfulness', 'answer_relevancy']:
            if col in df_ragas.columns:
                print(f"  {col:25s}: {df_ragas[col].mean():.4f} "
                      f"(min {df_ragas[col].min():.3f} / "
                      f"max {df_ragas[col].max():.3f})")
        print("-"*60)

        df_ragas['question'] = q_list
        df_ragas.to_csv("outputs/ragas_results.csv", index=False)
        print("✅ Saved → outputs/ragas_results.csv")
        return True, df_ragas

    except Exception as e:
        print(f"\n⚠️ RAGAS failed: {e}")
        print("Continuing with functional evaluation only.")
        return False, None


# ── PART 3: LLM-as-Judge Comparison ──────────────────────
def run_llm_judge_comparison(model, index, chunks, client):
    """
    Compares Simple RAG vs Agentic RAG on 5 questions.
    LLM judges each answer on 4 criteria (1-5 scale), grounded
    against the actual retrieved evidence for that answer:
    - Relevance
    - Groundedness
    - Citation support
    - Completeness
    Reports mean scores and % difference.
    """
    from phase4_agent import react_agent

    comparison_questions = [
        "How does RAG reduce hallucination and how is this measured?",
        "What limitations do RAG papers identify and what future work is proposed?",
        "Compare SBERT with standard BERT for semantic search in RAG systems",
        "How does ReAct framework improve over simple RAG for multi-hop questions?",
        "What is the relationship between RAGAS metrics and answer quality?",
    ]

    criteria = ["relevance", "groundedness", "citation_support", "completeness"]

    def judge_answer(question, answer, answer_type, evidence_text=""):
        """
        Ask Groq LLM to judge answer quality 1-5 on 4 criteria,
        grounded against the retrieved evidence for this answer.
        """
        prompt = f"""You are an expert evaluator for academic AI research assistants.

Rate this {answer_type} answer on these 4 criteria (score 1-5 each):
1. Relevance: Does it directly answer the question?
2. Groundedness: Is it based on the evidence below, not hallucination?
3. Citation support: Does it cite specific papers/sources that actually appear in the evidence?
4. Completeness: Does it cover all key aspects?

Question: {question}

Retrieved evidence the answer was supposed to be grounded in:
{evidence_text[:1200] if evidence_text else "(no evidence captured for this run)"}

Answer: {answer[:500]}

Reply ONLY with valid JSON containing ALL FOUR keys, e.g.:
{{"relevance": 4, "groundedness": 5, "citation_support": 3, "completeness": 4}}
Where each value is an integer 1-5. No other text."""

        try:
            import groq
            gc = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = gc.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            text = resp.choices[0].message.content.strip()
            nums = re.findall(r'"(\w+)":\s*(\d)', text)
            scores = {k: int(v) for k, v in nums}

            # FIX: require ALL four criteria to be present.
            # A partial match must NOT silently produce a
            # deflated "valid" mean — that corrupts the average
            # exactly like the fake-3/5-score bug this file
            # already tries to avoid on the exception path.
            if not all(c in scores for c in criteria):
                print(f"    ⚠️ Judge returned incomplete scores "
                      f"({len(scores)}/4 criteria) — treating as invalid")
                return None
            return scores
        except Exception as e:
            print(f"    Judge error: {e}")
            return None

    results = []

    print("\n" + "="*60)
    print("PART 3: LLM-AS-JUDGE COMPARISON")
    print("Simple RAG vs Agentic RAG (5 Questions)")
    print("="*60)

    for i, q in enumerate(comparison_questions):
        print(f"\n[{i+1}/5] {q[:55]}...")

        # Simple RAG
        s_ans, s_ctx, s_src = "Could not generate answer.", [], []
        try:
            s_ans, s_ctx, s_src = safe_rag(
                q, model, index, chunks, client, top_k=2
            )
            print(f"  Simple: {len(s_src)} sources, {len(s_ans)} chars ✅")
        except Exception as e:
            print(f"  Simple error: {str(e)[:40]}")
        time.sleep(8)
        s_evidence = " | ".join(s_ctx)

        # Agentic RAG
        a_ans, a_src, a_steps = "Could not generate answer.", [], 0
        a_evidence = ""
        try:
            r = react_agent(q, model, index, chunks, client,
                          max_searches=2)
            a_ans  = r['answer']
            a_src  = r['sources']
            a_steps = r['steps_taken']
            # FIX: use the exact evidence Phase 4 actually synthesised
            # from (pooled/deduped/garbled-filtered/capped), not the
            # per-search reasoning_trace observations, which were only
            # intermediate LLM summaries and didn't reflect what
            # informed the final answer. This makes "groundedness" and
            # "citation_support" judged against real final evidence.
            a_evidence = "\n\n".join(
                f"Source: {ev.get('source', 'Unknown')}\n"
                f"Evidence: {ev.get('text', '')}"
                for ev in r.get("evidence", [])
            )
            print(f"  Agent:  {len(a_src)} sources, "
                  f"{len(a_ans)} chars, {a_steps} steps ✅")
        except Exception as e:
            print(f"  Agent error: {str(e)[:40]}")
        time.sleep(10)

        # Judge both answers — now grounded against real evidence
        print("  Judging answers...")
        s_scores = judge_answer(q, s_ans, "Simple RAG", s_evidence)
        time.sleep(5)
        a_scores = judge_answer(q, a_ans, "Agentic RAG", a_evidence)
        time.sleep(5)

        # Handle None scores — exclude from mean.
        # None = judge failed or returned incomplete data —
        # never a real score of 0 or 3.
        s_mean = (sum(s_scores.values()) / len(criteria)
                  if s_scores is not None else None)
        a_mean = (sum(a_scores.values()) / len(criteria)
                  if a_scores is not None else None)

        if s_mean is not None and a_mean is not None and s_mean > 0:
            diff = (a_mean - s_mean) / s_mean * 100
            print(f"  Simple mean: {s_mean:.2f} | Agent mean: {a_mean:.2f} "
                  f"| Diff: {diff:+.1f}%")
        elif s_mean is None or a_mean is None:
            diff = None
            print(f"  ⚠️ Judge failed for this question — excluded from mean")
        else:
            diff = 0.0
            print(f"  Simple mean: {s_mean:.2f} | Agent mean: {a_mean:.2f}")

        row = {
            "question"          : q,
            "simple_answer"     : s_ans[:200],
            "simple_evidence"   : s_evidence[:500],
            "simple_sources"    : len(s_src),
            "simple_length"     : len(s_ans),
            "agent_answer"      : a_ans[:200],
            "agent_evidence"    : a_evidence[:500],
            "agent_sources"     : len(a_src),
            "agent_steps"       : a_steps,
            "agent_longer"      : len(a_ans) > len(s_ans),
            "agent_more_sources": len(a_src) > len(s_src),
            "judge_valid"       : s_mean is not None and a_mean is not None,
        }
        for crit in criteria:
            row[f"simple_{crit}"] = (s_scores.get(crit)
                                     if s_scores is not None else None)
            row[f"agent_{crit}"]  = (a_scores.get(crit)
                                     if a_scores is not None else None)
        row["simple_mean"]    = round(s_mean, 3) if s_mean is not None else None
        row["agent_mean"]     = round(a_mean, 3) if a_mean is not None else None
        row["pct_difference"] = round(diff, 1)   if diff  is not None else None
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("outputs/comparison_results.csv", index=False)

    # Summary — only include valid (non-None) judge scores in mean
    valid = [r for r in results if r["judge_valid"]]
    invalid_count = len(results) - len(valid)

    if valid:
        s_overall = sum(r["simple_mean"] for r in valid) / len(valid)
        a_overall = sum(r["agent_mean"]  for r in valid) / len(valid)
        overall_diff = ((a_overall - s_overall) / s_overall * 100
                        if s_overall > 0 else 0)
    else:
        s_overall = a_overall = overall_diff = 0

    print("\n" + "-"*60)
    print("LLM-AS-JUDGE RESULTS — TABLE 3")
    print("-"*60)
    print(f"  Questions compared     : {len(results)}")
    print(f"  Valid judge responses  : {len(valid)}/{len(results)}")
    if invalid_count > 0:
        print(f"  ⚠️ {invalid_count} question(s) excluded — judge returned "
              f"incomplete/invalid data")
    if valid:
        print(f"  Simple RAG overall     : {s_overall:.3f} / 5.0")
        print(f"  Agentic RAG overall    : {a_overall:.3f} / 5.0")
        print(f"  % Difference           : {overall_diff:+.1f}%")
    else:
        print("  ⚠️ No valid judge scores — check GROQ_API_KEY")
    print(f"  Agent longer answer    : "
          f"{sum(r['agent_longer'] for r in results)}/5")
    print(f"  Agent more sources     : "
          f"{sum(r['agent_more_sources'] for r in results)}/5")
    print(f"  Avg agent steps        : "
          f"{sum(r['agent_steps'] for r in results)/len(results):.1f}")
    print("-"*60)
    print("  NOTE: Report the actual % difference from YOUR run.")
    print("  Do not select the nicer result — use the frozen run.")
    print("-"*60)
    print("✅ Saved → outputs/comparison_results.csv")
    return df


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    from phase2_embed import load_embedding_model, load_index
    from phase3_rag import setup_llm

    print("="*60)
    print("PHASE 5 — COMPLETE EVALUATION")
    print("ResearchForge | Piyush Bhati | SIG 2025-26")
    print("="*60)
    print("THREE PARTS:")
    print("  Part 1: Functional evaluation  (20 questions)")
    print("  Part 2: RAGAS automated scoring")
    print("  Part 3: LLM-as-Judge comparison (5 questions)")
    print("="*60)

    model         = load_embedding_model()
    index, chunks = load_index()
    client        = setup_llm()

    print(f"Loaded: {len(chunks)} chunks, SBERT ready\n")

    # ── Part 1 ────────────────────────────────────────────
    questions = get_test_questions()
    q_list, a_list, c_list = run_functional_eval(
        questions, model, index, chunks, client
    )

    # Save raw answers
    with open("outputs/raw_answers.json", "w") as f:
        json.dump(
            [{"q": q, "a": a} for q, a in zip(q_list, a_list)],
            f, indent=2
        )
    print("✅ Raw answers → outputs/raw_answers.json")

    # ── Part 2 ────────────────────────────────────────────
    ragas_ok, df_ragas = run_ragas_eval(q_list, a_list, c_list)

    # ── Part 3 ────────────────────────────────────────────
    df_compare = run_llm_judge_comparison(model, index, chunks, client)

    # ── Final Summary ─────────────────────────────────────
    # All numbers below are computed from the actual run above —
    # nothing here is a placeholder or a hand-picked result.
    answered_n  = sum(1 for a in a_list if "could not" not in a.lower())
    avg_len     = sum(len(a) for a in a_list) / len(a_list)
    avg_ctx     = sum(len(c) for c in c_list) / len(c_list)

    criteria = ["relevance", "groundedness", "citation_support", "completeness"]
    valid_rows = df_compare[df_compare["judge_valid"] == True] if not df_compare.empty else df_compare

    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)

    print("\nFUNCTIONAL EVALUATION")
    print(f"Questions tested       : {len(q_list)}")
    print(f"Questions answered     : {answered_n}")
    print(f"Operational completion : {answered_n/len(q_list)*100:.1f}%")
    print(f"Average answer length  : {avg_len:.0f}")
    print(f"Average contexts       : {avg_ctx:.2f}")

    print("\nRAGAS")
    if ragas_ok and df_ragas is not None:
        faith = df_ragas['faithfulness'].mean() if 'faithfulness' in df_ragas else float('nan')
        rel   = df_ragas['answer_relevancy'].mean() if 'answer_relevancy' in df_ragas else float('nan')
        import math
        print(f"Faithfulness           : {'Not available (NaN)' if math.isnan(faith) else f'{faith:.4f}'}")
        print(f"Answer Relevancy       : {'Not available (NaN)' if math.isnan(rel) else f'{rel:.4f}'}")
        print(f"Status                 : Completed" if not (math.isnan(faith) and math.isnan(rel)) else "Status                 : Partial")
    else:
        print("Faithfulness           : Not available")
        print("Answer Relevancy       : Not available")
        print("Status                 : Failed")

    print("\nSIMPLE RAG vs AGENTIC RAG")
    print(f"Valid comparisons      : {len(valid_rows)}/{len(df_compare)}")
    if len(valid_rows) > 0:
        print(f"\n{'':25s}{'Simple':>10s}{'Agentic':>10s}")
        for crit in criteria:
            s_col, a_col = f"simple_{crit}", f"agent_{crit}"
            s_avg = valid_rows[s_col].mean()
            a_avg = valid_rows[a_col].mean()
            label = crit.replace("_", " ").title()
            print(f"{label:25s}{s_avg:10.2f}{a_avg:10.2f}")
        s_overall = valid_rows["simple_mean"].mean()
        a_overall = valid_rows["agent_mean"].mean()
        overall_diff = ((a_overall - s_overall) / s_overall * 100) if s_overall > 0 else 0.0
        print(f"{'Overall':25s}{s_overall:10.2f}{a_overall:10.2f}")
        print(f"\nAgentic difference      : {overall_diff:+.1f}%")
    else:
        print("⚠️ No valid judge comparisons — check GROQ_API_KEY / judge output")

    print(f"Agent longer answers    : {int(df_compare['agent_longer'].sum())}/{len(df_compare)}")
    print(f"Agent more sources      : {int(df_compare['agent_more_sources'].sum())}/{len(df_compare)}")
    print(f"Average agent steps     : {df_compare['agent_steps'].mean():.2f}")

    print("\n" + "="*60)
    print("PHASE 5 COMPLETE — OUTPUT FILES")
    print("="*60)
    print("  outputs/raw_answers.json        — 20 raw answers")
    print("  outputs/functional_results.csv  — Table 1: operational")
    if ragas_ok:
        print("  outputs/ragas_results.csv       — Table 2: RAGAS scores")
    else:
        print("  outputs/ragas_results.csv       — NOT generated (see note)")
    print("  outputs/comparison_results.csv  — Table 3: judge scores "
          "(now includes simple_evidence / agent_evidence columns)")
    print("="*60)
    print("\nIMPORTANT: Use actual numbers from YOUR run — do not")
    print("select the nicer result. Freeze after one successful run.")
    print("="*60)