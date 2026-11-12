
import asyncio

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_faithfulness,
    evaluate_answer_relevancy
)


async def main():

    # Step 1: Build evaluation dataset
    print("\nBUILDING EVALUATION DATASET...")

    dataset = build_evaluation_dataset()

    print("\nTotal Questions:", len(dataset))


    # Step 2: Evaluate Faithfulness
    print("\n========== FAITHFULNESS EVALUATION ==========")

    faithfulness_scores = await evaluate_faithfulness(dataset)


    # Step 3: Evaluate Answer Relevancy
    print("\n========== ANSWER RELEVANCY EVALUATION ==========")

    relevancy_scores = await evaluate_answer_relevancy(dataset)


    # Step 4: Calculate average scores
    average_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
    )

    average_relevancy = (
        sum(relevancy_scores) / len(relevancy_scores)
    )


    # Step 5: Display final evaluation summary
    print("\n========== FINAL EVALUATION REPORT ==========")

    print("Total Questions:", len(dataset))

    print("\nFaithfulness Scores:", faithfulness_scores)
    print("Average Faithfulness:", round(average_faithfulness, 3))

    print("\nAnswer Relevancy Scores:", relevancy_scores)
    print("Average Answer Relevancy:", round(average_relevancy, 3))

    print("\n=============================================")


asyncio.run(main())
