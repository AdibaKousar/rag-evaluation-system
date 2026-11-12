import asyncio

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_faithfulness
)


async def main():

    # Step 1: Build evaluation dataset
    print("\nBUILDING EVALUATION DATASET...")

    dataset = build_evaluation_dataset()

    print("\nTotal questions:", len(dataset))

    # Step 2: Evaluate Faithfulness
    print("\nSTARTING FAITHFULNESS EVALUATION...")

    scores = await evaluate_faithfulness(dataset)

    # Step 3: Calculate average score
    average_score = sum(scores) / len(scores)

    print("\n========== EVALUATION SUMMARY ==========")

    print("Total Questions:", len(scores))
    print("Individual Scores:", scores)
    print("Average Faithfulness:", round(average_score, 3))

    print("========================================")


asyncio.run(main())