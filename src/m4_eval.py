from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ragas_llm_and_embeddings():
    """Build RAGAS LLM/embedding wrappers pointed at LM Studio (OpenAI-compatible).

    ⚠️ LƯU Ý: không dùng llm_factory()/embedding_factory() mặc định của RAGAS —
    chúng hard-code `gpt-4o-mini` và `text-embedding-ada-002`, hai model mà
    LM Studio không phục vụ. ChatOpenAI/OpenAIEmbeddings tự đọc OPENAI_API_BASE
    từ .env, nên chúng trỏ về http://127.0.0.1:1234/v1.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from config import LLM_MODEL, RAGAS_EMBEDDING_MODEL

    return (
        LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL, temperature=0, max_retries=3)),
        LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL)),
    )


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    # RAGAS cần OPENAI_API_KEY, Python 3.11+ và rất nhiều LLM call → bọc try/except
    # để một câu hỏi hỏng không làm sập cả batch.
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                    context_recall, faithfulness)

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        llm, embeddings = _ragas_llm_and_embeddings()
        result = evaluate(dataset,
                          metrics=[faithfulness, answer_relevancy,
                                   context_precision, context_recall],
                          llm=llm, embeddings=embeddings, raise_exceptions=False)

        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=row["contexts"], ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness") or 0.0),
                answer_relevancy=float(row.get("answer_relevancy") or 0.0),
                context_precision=float(row.get("context_precision") or 0.0),
                context_recall=float(row.get("context_recall") or 0.0),
            )
            for _, row in df.iterrows()
        ]

        def avg(attr: str) -> float:
            if not per_question:
                return 0.0
            return round(sum(getattr(r, attr) for r in per_question) / len(per_question), 4)

        return {"faithfulness": avg("faithfulness"),
                "answer_relevancy": avg("answer_relevancy"),
                "context_precision": avg("context_precision"),
                "context_recall": avg("context_recall"),
                "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM đang hallucinate — câu trả lời có thông tin không có trong context",
                         "Siết prompt, hạ temperature xuống 0, ép trích dẫn nguồn"),
        "context_recall": ("Thiếu chunk chứa đáp án trong context đã truy xuất",
                           "Cải thiện chunking, thêm BM25 vào hybrid, nới top_k khi truy xuất"),
        "context_precision": ("Quá nhiều chunk không liên quan lọt vào top-k",
                              "Thêm reranking, lọc theo metadata, giảm top_k xuống 3"),
        "answer_relevancy": ("Câu trả lời không khớp câu hỏi",
                             "Cải thiện prompt template, ép trả lời đúng 1 ý chính"),
    }

    scored = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=lambda k: metrics[k])
        avg_score = sum(metrics.values()) / 4
        scored.append((avg_score, result, worst_metric))

    # Sắp xếp tăng dần theo điểm trung bình → lấy bottom_n câu hỏi tệ nhất.
    scored.sort(key=lambda item: item[0])

    failures = []
    for avg_score, result, worst_metric in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
