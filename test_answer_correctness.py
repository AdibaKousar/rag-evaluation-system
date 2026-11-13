
import asyncio

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_answer_correctness
)


async def main():

    print("\nBUILDING EVALUATION DATASET...")

    dataset = build_evaluation_dataset()

    print("\nTotal Questions:", len(dataset))

    print("\n========== ANSWER CORRECTNESS EVALUATION ==========")

    scores = await evaluate_answer_correctness(dataset)

    average_score = sum(scores) / len(scores)

    print("\n========== ANSWER CORRECTNESS SUMMARY ==========")

    print("Total Questions:", len(scores))

    print("Individual Scores:", scores)

    print("Average Answer Correctness:", round(average_score, 3))

    print("===============================================")


asyncio.run(main())
