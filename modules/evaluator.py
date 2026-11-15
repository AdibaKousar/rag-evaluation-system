import csv
import os
from datetime import datetime
import json

from openai import AsyncOpenAI

from ragas import EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness
)

from modules.rag_pipeline import generate_rag_answer_with_context


# ==========================================
# STEP 1: CONNECT TO LOCAL OLLAMA
# ==========================================

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# ==========================================
# STEP 2: INITIALIZE QWEN EVALUATOR
# ==========================================

evaluator_llm = llm_factory(
    "qwen2.5:1.5b-instruct",
    provider="openai",
    client=client
)


# ==========================================
# STEP 3: INITIALIZE EMBEDDING MODEL
# ==========================================

evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# STEP 4: INITIALIZE RAGAS METRICS
# ==========================================

faithfulness_metric = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

context_precision_metric = ContextPrecision(
    llm=evaluator_llm
)

context_recall_metric = ContextRecall(
    llm=evaluator_llm
)

answer_correctness_metric = AnswerCorrectness(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


# ==========================================
# STEP 5: BUILD EVALUATION DATASET
# ==========================================


def build_evaluation_dataset(document_id):

    with open(
        "data/evaluation_questions.json",
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    evaluation_samples = []

    for item in questions:

        question = item["question"]
        reference = item["reference"]

        print("\nProcessing question:", question)

        result = generate_rag_answer_with_context(
            question,
            document_id
        )

        contexts = result.get("retrieved_contexts") or []

        if isinstance(contexts, str):
            contexts = [contexts]

        contexts = [
            str(context)
            for context in contexts
            if context and str(context).strip()
        ]

        if not contexts:
            contexts = [
                "No relevant information was retrieved from the document."
            ]

        sample = {
            "user_input": question,
            "response": result.get("answer", ""),
            "retrieved_contexts": contexts,
            "reference": reference
        }

        evaluation_samples.append(sample)

    dataset = EvaluationDataset.from_list(
        evaluation_samples
    )

    print("\nDataset created:", len(dataset), "samples")

    return dataset

# ==========================================
# STEP 6: EVALUATE FAITHFULNESS
# ==========================================


async def evaluate_faithfulness(dataset):

    scores = []

    for index, sample in enumerate(dataset):

        print(
            f"\nEvaluating Faithfulness: "
            f"{index + 1}/{len(dataset)}"
        )

        contexts = sample.retrieved_contexts or []

        if isinstance(contexts, str):
            contexts = [contexts]

        result = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=contexts
        )

        scores.append(result.value)

        print("Faithfulness Score:", result.value)

    return scores


# ==========================================
# STEP 7: EVALUATE ANSWER RELEVANCY
# ==========================================

async def evaluate_answer_relevancy(dataset):

    """Check whether answers are relevant to the questions."""

    scores = []

    for index, sample in enumerate(dataset):

        print(f"\nEvaluating Answer Relevancy: {index + 1}/{len(dataset)}")

        result = await answer_relevancy_metric.ascore(
            user_input=sample.user_input,
            response=sample.response
        )

        scores.append(result.value)

        print("Answer Relevancy Score:", result.value)

    return scores


# ==========================================
# STEP 8: EVALUATE CONTEXT PRECISION
# ==========================================

async def evaluate_context_precision(dataset):

    """Check whether relevant contexts are ranked higher."""

    scores = []

    for index, sample in enumerate(dataset):

        print(f"\nEvaluating Context Precision: {index + 1}/{len(dataset)}")

        result = await context_precision_metric.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts
        )

        scores.append(result.value)

        print("Context Precision Score:", result.value)

    return scores


# ==========================================
# STEP 9: EVALUATE CONTEXT RECALL
# ==========================================

async def evaluate_context_recall(dataset):

    """Check whether retrieved contexts contain reference information."""

    scores = []

    for index, sample in enumerate(dataset):

        print(f"\nEvaluating Context Recall: {index + 1}/{len(dataset)}")

        result = await context_recall_metric.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts
        )

        scores.append(result.value)

        print("Context Recall Score:", result.value)

    return scores


# ==========================================
# STEP 10: EVALUATE ANSWER CORRECTNESS
# ==========================================

async def evaluate_answer_correctness(dataset):

    """Compare generated answers with reference answers."""

    scores = []

    for index, sample in enumerate(dataset):

        print(f"\nEvaluating Answer Correctness: {index + 1}/{len(dataset)}")

        result = await answer_correctness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            reference=sample.reference
        )

        scores.append(result.value)

        print("Answer Correctness Score:", result.value)

    return scores


def save_evaluation_report(results):

    """
    Save all evaluation metric scores into JSON and CSV files.
    """

    # Create reports folder if it does not exist
    os.makedirs("reports", exist_ok=True)

    # Calculate average scores
    report = {}

    for metric_name, scores in results.items():

        average = sum(scores) / len(scores) if scores else 0

        report[metric_name] = {
            "individual_scores": scores,
            "average_score": round(average, 3)
        }

    # Save JSON report
    json_path = "reports/evaluation_report.json"

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    # Save CSV report
    csv_path = "reports/evaluation_report.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Metric",
            "Question Number",
            "Score"
        ])

        for metric_name, scores in results.items():

            for index, score in enumerate(scores, start=1):

                writer.writerow([
                    metric_name,
                    index,
                    score
                ])

    print("\nEvaluation reports saved successfully!")

    print("JSON Report:", json_path)

    print("CSV Report:", csv_path)
