
import asyncio

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_answer_correctness
)


async def main():

    print("\n========== BUILDING EVALUATION DATASET ==========")

    dataset = build_evaluation_dataset()

    print("\nTotal Questions:", len(dataset))

    # Evaluate all five metrics
    print("\n========== EVALUATING ALL FIVE METRICS ==========")

    faithfulness_scores = await evaluate_faithfulness(dataset)

    answer_relevancy_scores = await evaluate_answer_relevancy(dataset)

    context_precision_scores = await evaluate_context_precision(dataset)

    context_recall_scores = await evaluate_context_recall(dataset)

    answer_correctness_scores = await evaluate_answer_correctness(dataset)

    # Calculate averages
    results = {
        "Faithfulness": faithfulness_scores,
        "Answer Relevancy": answer_relevancy_scores,
        "Context Precision": context_precision_scores,
        "Context Recall": context_recall_scores,
        "Answer Correctness": answer_correctness_scores
    }

    print("\n========== FINAL RAG EVALUATION REPORT ==========")

    for metric_name, scores in results.items():

        average = sum(scores) / len(scores)

        print(f"\n{metric_name}:")
        print("Individual Scores:", scores)
        print("Average Score:", round(average, 3))

    print("\n========== EVALUATION COMPLETED ==========")


asyncio.run(main())
