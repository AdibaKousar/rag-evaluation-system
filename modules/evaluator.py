
import json

from openai import AsyncOpenAI

from ragas import EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy
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


# ==========================================
# STEP 5: BUILD EVALUATION DATASET
# ==========================================

def build_evaluation_dataset():

    """
    Generate answers for all evaluation questions
    and create a RAGAS evaluation dataset.
    """

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

        result = generate_rag_answer_with_context(question)

        sample = {
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": result["retrieved_contexts"],
            "reference": reference
        }

        evaluation_samples.append(sample)

    dataset = EvaluationDataset.from_list(
        evaluation_samples
    )

    return dataset


# ==========================================
# STEP 6: EVALUATE FAITHFULNESS
# ==========================================

async def evaluate_faithfulness(dataset):

    """
    Check whether generated answers are supported
    by the retrieved contexts.
    """

    scores = []

    for index, sample in enumerate(dataset):

        print(
            f"\nEvaluating Faithfulness: "
            f"{index + 1}/{len(dataset)}"
        )

        result = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts
        )

        score = result.value

        scores.append(score)

        print("Faithfulness Score:", score)

    return scores


# ==========================================
# STEP 7: EVALUATE ANSWER RELEVANCY
# ==========================================

async def evaluate_answer_relevancy(dataset):

    """
    Check whether generated answers are relevant
    to the original questions.
    """

    scores = []

    for index, sample in enumerate(dataset):

        print(
            f"\nEvaluating Answer Relevancy: "
            f"{index + 1}/{len(dataset)}"
        )

        result = await answer_relevancy_metric.ascore(
            user_input=sample.user_input,
            response=sample.response
        )

        score = result.value

        scores.append(score)

        print("Answer Relevancy Score:", score)

    return scores
