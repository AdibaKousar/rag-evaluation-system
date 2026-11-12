
import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


async def main():

    # Step 1: Connect to local TinyLlama
    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )

    # Step 2: Create RAGAS evaluator
    evaluator_llm = llm_factory(
    "qwen2.5:1.5b-instruct",
    provider="openai",
    client=client
)

    # Step 3: Initialize Faithfulness metric
    faithfulness = Faithfulness(llm=evaluator_llm)

    # Step 4: Evaluate a sample answer
    result = await faithfulness.ascore(
        user_input="What is machine learning?",

        response=(
    "Machine Learning is a subset of Artificial "
    "Intelligence that allows computers to learn "
    "from data."
),

        retrieved_contexts=[
            "Machine Learning is a subset of Artificial "
            "Intelligence. It allows computers to learn "
            "from data and improve their performance "
            "without being explicitly programmed."
        ]
    )

    # Step 5: Display score
    print("\nFAITHFULNESS SCORE:")
    print(result.value)

    print("\nEVALUATION REASON:")
    print(result.reason)


asyncio.run(main())
