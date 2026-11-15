# RAG Evaluation System

## Project Overview
The RAG Evaluation System is an AI-based application that allows users to upload PDF documents, ask questions about their content, and evaluate the quality of AI-generated answers.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate answers. It also evaluates the answers using five evaluation metrics.

## Features
- Upload and process PDF documents.
- Extract and split document text into smaller chunks.
- Generate text embeddings using Sentence Transformers.
- Store and retrieve information using ChromaDB.
- Generate answers using Ollama.
- Evaluate AI-generated answers using five RAGAS metrics.
- Display evaluation scores through an interactive Streamlit dashboard.
- Download evaluation reports in JSON and CSV formats.
- Support multiple PDF documents.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and dashboard |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| Ollama | Local AI answer generation |
| RAGAS | RAG evaluation |
| PyPDF | PDF text extraction |
| Pandas | Report processing |

## Five Evaluation Metrics

1. **Faithfulness:** Checks whether the answer is supported by the retrieved information.
2. **Answer Relevancy:** Checks whether the answer is relevant to the question.
3. **Context Precision:** Measures how relevant the retrieved information is.
4. **Context Recall:** Checks whether the retrieved information covers the required reference information.
5. **Answer Correctness:** Measures how correct the generated answer is compared with the reference answer.

## Project Workflow

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Generate embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant information when a question is asked.
7. Generate an answer using Ollama.
8. Evaluate the answer using RAGAS.
9. Display results on the Streamlit dashboard.
10. Download evaluation reports.

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-evaluation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

Make sure Ollama is installed and the required models are available.

```bash
ollama serve
```

### 4. Run the application

Open another terminal and execute:

```bash
streamlit run app.py
```

## Evaluation Results

The project evaluates 20 questions using five RAGAS metrics.

The evaluation results are displayed in the dashboard and can be downloaded in JSON and CSV formats.

*Note: Evaluation scores depend on the document, questions, models, and retrieval quality.*

## Project Structure

```text
rag-evaluation-system/
│
├── data/
│   └── evaluation_questions.json
│
├── documents/
│   └── sample.pdf
│
├── modules/
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── evaluator.py
│
├── reports/
│   ├── evaluation_report.json
│   └── evaluation_report.csv
│
├── app.py
├── test_combined_evaluation.py
├── requirements.txt
└── README.md
```

## Future Improvements
- Improve retrieval accuracy.
- Support additional document formats.
- Improve answer generation quality.
- Add more evaluation metrics.
- Improve dashboard visualizations.

## Conclusion

This project demonstrates how Retrieval-Augmented Generation can be used to build a document-based question-answering system and how RAGAS can help evaluate the quality of its responses.