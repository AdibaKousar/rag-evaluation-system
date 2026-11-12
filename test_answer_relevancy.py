from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import AnswerRelevancy


# Step 1: Connect to local Qwen evaluator
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# Step 2: Initialize Qwen
evaluator_llm = llm_factory(
    "qwen2.5:1.5b-instruct",
    provider="openai",
    client=client
)


# Step 3: Initialize embedding model
evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Initialize Answer Relevancy metric
answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


import asyncio


async def main():

    result = await answer_relevancy.ascore(
        user_input="What is machine learning?",

        response=(
    "Machine Learning is a subset of Artificial "
    "Intelligence that allows computers to learn "
    "from data and improve their performance."
)
    )

    print("\nANSWER RELEVANCY SCORE:")
    print(result.value)

    print("\nEVALUATION REASON:")
    print(result.reason)


asyncio.run(main())