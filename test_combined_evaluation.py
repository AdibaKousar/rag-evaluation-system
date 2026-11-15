
import asyncio

from modules.retriever import get_uploaded_documents

from modules.evaluator import (
    build_evaluation_dataset,
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_answer_correctness,
    save_evaluation_report
)


async def main():

    print("\n========== BUILDING EVALUATION DATASET ==========")

    # Get all indexed documents
    uploaded_documents = get_uploaded_documents()

    # Find the dedicated AI evaluation document
    ai_documents = {
        doc_id: name
        for doc_id, name in uploaded_documents.items()
        if name.lower() == "ai_evaluation_document.pdf"
    }

    if not ai_documents:

        print(
            "AI_Evaluation_Document.pdf not found. "
            "Please upload and index the AI PDF first."
        )

        return

    # Use the known evaluation document ID
    document_id = "a8f3a8ea-b4a5-4ac2-84c0-3d285369e1b5"

    if document_id not in ai_documents:

        print("The specified evaluation document ID was not found.")
        print("Available evaluation document IDs:")

        for doc_id, name in ai_documents.items():
            print(name, "-", doc_id)

        return

    print("\nSelected AI evaluation document:")
    print(ai_documents[document_id])
    print("Document ID:", document_id)

    # Build evaluation dataset
    dataset = build_evaluation_dataset(document_id)

    print("\nDataset loaded successfully!")
    print("Total evaluation questions:", len(dataset))

    if len(dataset) != 20:

        print(
            "\nWARNING: Expected 20 questions, "
            f"but found {len(dataset)}."
        )

        return

    print("\nAll 20 questions are ready for evaluation.")

    # Evaluate all five metrics
    print("\n========== EVALUATING ALL FIVE METRICS ==========")

    faithfulness_scores = await evaluate_faithfulness(dataset)

    answer_relevancy_scores = await evaluate_answer_relevancy(dataset)

    context_precision_scores = await evaluate_context_precision(dataset)

    context_recall_scores = await evaluate_context_recall(dataset)

    answer_correctness_scores = await evaluate_answer_correctness(dataset)

    # Collect results
    results = {
        "Faithfulness": faithfulness_scores,
        "Answer Relevancy": answer_relevancy_scores,
        "Context Precision": context_precision_scores,
        "Context Recall": context_recall_scores,
        "Answer Correctness": answer_correctness_scores
    }

    # Save evaluation reports
    save_evaluation_report(results)

    # Display final results
    print("\n========== FINAL RAG EVALUATION REPORT ==========")

    for metric_name, scores in results.items():

        if scores:

            average = sum(scores) / len(scores)

            print(f"\n{metric_name}:")
            print("Individual Scores:", scores)
            print("Total Evaluated:", len(scores))
            print("Average Score:", round(average, 3))

        else:

            print(f"\n{metric_name}: No scores generated.")

    print("\n========== EVALUATION COMPLETED ==========")


asyncio.run(main())
