
import asyncio

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_context_precision
)


async def main():

    print("\nBUILDING EVALUATION DATASET...")

    dataset = build_evaluation_dataset()

    print("\nTotal Questions:", len(dataset))

    print("\n========== CONTEXT PRECISION EVALUATION ==========")

    scores = await evaluate_context_precision(dataset)

    average_score = sum(scores) / len(scores)

    print("\n========== CONTEXT PRECISION SUMMARY ==========")

    print("Total Questions:", len(scores))

    print("Individual Scores:", scores)

    print("Average Context Precision:", round(average_score, 3))

    print("===============================================")


asyncio.run(main())
