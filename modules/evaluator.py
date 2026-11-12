import json
import asyncio

from openai import AsyncOpenAI
from ragas import EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from modules.rag_pipeline import generate_rag_answer_with_context


# Connect to local Ollama
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# Create RAGAS evaluator LLM using Qwen
evaluator_llm = llm_factory(
    "qwen2.5:1.5b-instruct",
    provider="openai",
    client=client
)


# Initialize Faithfulness metric
faithfulness_metric = Faithfulness(llm=evaluator_llm)


def build_evaluation_dataset():
    """
    Generate answers for all evaluation questions
    and create a RAGAS evaluation dataset.
    """

    with open("data/evaluation_questions.json", "r", encoding="utf-8") as file:
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

    dataset = EvaluationDataset.from_list(evaluation_samples)

    return dataset


async def evaluate_faithfulness(dataset):
    """
    Evaluate faithfulness for every sample in the dataset.
    """

    scores = []

    for index, sample in enumerate(dataset):

        print(f"\nEvaluating Faithfulness: {index + 1}/{len(dataset)}")

        result = await faithfulness_metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts
        )

        score = result.value

        scores.append(score)

        print("Faithfulness Score:", score)

    return scores