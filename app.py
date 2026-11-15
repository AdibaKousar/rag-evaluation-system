
import streamlit as st
import plotly.express as px
import pandas as pd
import json
from pathlib import Path
import tempfile
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Intelligence Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F6F5FA;
    color: #241B35;
}

.stApp p,
.stApp label,
.stApp li,
.stApp [data-testid="stMarkdownContainer"] {
    color: #302443;
}

h1, h2, h3, h4 {
    color: #35245C !important;
    font-weight: 750 !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #211B35 !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    opacity: 1;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #FFFFFF !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5DDF2;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #514568 !important;
}

[data-testid="stMetricValue"] {
    color: #6843B8 !important;
}

/* CUSTOM CARDS */

.metric-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E8E3F2;
    box-shadow: 0 3px 12px rgba(45, 30, 75, 0.05);
}

.metric-label {
    color: #514568 !important;
    font-size: 14px;
}

.metric-value {
    color: #6843B8 !important;
    font-size: 29px;
    font-weight: 800;
}

.section-title {
    color: #35245C !important;
    font-size: 22px;
    font-weight: 700;
}

/* INPUTS */

.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #241B35 !important;
    border: 1px solid #B8A9D0 !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF;
    color: #241B35 !important;
}

/* BUTTONS */

.stApp div.stButton > button,
.stApp div.stDownloadButton > button {
    background-color: #7653C6 !important;
    color: #FFFFFF !important;
    border: 1px solid #7653C6 !important;
    border-radius: 9px;
    font-weight: 600 !important;
}

.stApp div.stButton > button *,
.stApp div.stDownloadButton > button * {
    color: #FFFFFF !important;
}

.stApp div.stButton > button:hover,
.stApp div.stDownloadButton > button:hover {
    background-color: #5938A5 !important;
    border-color: #5938A5 !important;
}

/* FILE UPLOADER */

.stApp [data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D8CBEA !important;
    border-radius: 12px;
    padding: 15px;
}

.stApp [data-testid="stFileUploader"] * {
    color: #302443 !important;
}

.stApp [data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
}

/* ALERTS */

.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] span,
.stApp [data-testid="stAlert"] div {
    color: #FFFFFF !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# REPORT LOCATION
# ==========================================

BASE_DIR = Path(__file__).parent

REPORT_PATH = BASE_DIR / "reports" / "evaluation_report.json"


# ==========================================
# LOAD EVALUATION REPORT
# ==========================================

def load_report():

    if REPORT_PATH.exists():

        try:
            with open(
                REPORT_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (json.JSONDecodeError, OSError) as error:

            st.error(f"Unable to load evaluation report: {error}")

    return {}


report = load_report()


# ==========================================
# RAG FUNCTION
# ==========================================

@st.cache_resource
def get_rag_function():

    from modules.rag_pipeline import (
        generate_rag_answer_with_context
    )

    return generate_rag_answer_with_context


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🤖 RAG Intelligence")

st.sidebar.caption(
    "Evaluation & Analytics Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Ask Questions",
        "Evaluation Results",
        "Upload PDF"
    ]
)

st.sidebar.divider()

st.sidebar.success("System Interface Active")

st.sidebar.caption(
    "Powered by Streamlit, ChromaDB and RAGAS"
)


# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">'
        'Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor your RAG system performance and '
        'evaluation metrics.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # Get indexed PDF documents
    from modules.retriever import get_uploaded_documents

    uploaded_documents = get_uploaded_documents()

    total_documents = len(uploaded_documents)

    if report:

        total_questions = max(
            (
                len(data.get("individual_scores", []))
                for data in report.values()
            ),
            default=0
        )

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Evaluation Metrics
                    </div>
                    <div class="metric-value">
                        {len(report)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Evaluated Questions
                    </div>
                    <div class="metric-value">
                        {total_questions}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">
                        Embedding Model
                    </div>
                    <div class="metric-value"
                         style="font-size:23px">
                        MiniLM
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Indexed PDF Uploads
                    </div>
                    <div class="metric-value">
                        {total_documents}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.divider()

        # Evaluation performance
        st.markdown(
            '<div class="section-title">'
            'Evaluation Performance</div>',
            unsafe_allow_html=True
        )

        metric_data = []

        for name, data in report.items():

            average = data.get("average_score", 0)

            metric_data.append({
                "Metric": name,
                "Average Score": average
            })

        # Display metric cards
        for index, metric in enumerate(metric_data):

            if index % 3 == 0:
                columns = st.columns(3)

            with columns[index % 3]:

                st.metric(
                    label=metric["Metric"],
                    value=f'{metric["Average Score"] * 100:.1f}%'
                )

        st.write("")
        st.divider()

        # Performance chart
        chart_df = pd.DataFrame(metric_data)

        chart_df["Average Score"] = (
            chart_df["Average Score"] * 100
        )

        fig = px.bar(
            chart_df,
            x="Metric",
            y="Average Score",
            color="Average Score",
            color_continuous_scale=[
                "#D8C8F5",
                "#7653C6",
                "#39205F"
            ],
            range_y=[0, 100],
            title="RAGAS Evaluation Performance",
            text="Average Score"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font_color="#35245C",
            showlegend=False,
            title={
                "text": "RAGAS Evaluation Performance",
                "x": 0.5,
                "xanchor": "center"
            },
            yaxis={
                "title": "Score (%)",
                "range": [0, 100],
                "ticksuffix": "%"
            },
            xaxis={
                "title": "Evaluation Metrics"
            },
            height=450
        )

        fig.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.warning(
            "Evaluation report not found. "
            "Run your evaluation first."
        )

        st.info(
            f"Currently indexed PDF uploads: {total_documents}"
        )


# ==========================================
# ASK QUESTIONS PAGE
# ==========================================

elif page == "Ask Questions":

    st.markdown(
        '<div class="main-title">'
        'Ask Your RAG System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select a PDF document and ask questions based on its content."
    )

    from modules.retriever import get_uploaded_documents

    uploaded_documents = get_uploaded_documents()

    if uploaded_documents:

        document_options = {
            f"{name} ({doc_id[:8]})": doc_id
            for doc_id, name in uploaded_documents.items()
        }

        selected_document = st.selectbox(
            "Select PDF Document",
            list(document_options.keys())
        )

        selected_document_id = document_options[
            selected_document
        ]

        st.divider()

        question = st.text_input(
            "Enter your question",
            placeholder="Example: What is artificial intelligence?"
        )

        if st.button("Generate Answer"):

            if question.strip():

                try:

                    with st.spinner(
                        "Retrieving information and generating answer..."
                    ):

                        rag_function = get_rag_function()

                        result = rag_function(
                            question,
                            selected_document_id
                        )

                    st.success("Answer generated successfully!")

                    st.subheader("Generated Answer")

                    st.write(result["answer"])

                    st.subheader("Retrieved Context")

                    contexts = result.get(
                        "retrieved_contexts", []
                    )

                    if contexts:

                        for index, context in enumerate(
                            contexts,
                            start=1
                        ):

                            with st.expander(
                                f"Context {index}"
                            ):

                                st.write(context)

                    else:

                        st.info(
                            "No relevant document context was retrieved."
                        )

                except Exception as error:

                    st.error(
                        f"Unable to generate answer: {error}"
                    )

            else:

                st.warning("Please enter a question.")

    else:

        st.info(
            "No indexed PDF documents found. "
            "Please upload and process a PDF first."
        )


# ==========================================
# EVALUATION RESULTS PAGE
# ==========================================

elif page == "Evaluation Results":

    st.markdown(
        '<div class="main-title">'
        'Detailed Evaluation Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Analyze individual scores for every evaluation metric."
    )

    if report:

        selected_metric = st.selectbox(
            "Select Evaluation Metric",
            list(report.keys())
        )

        metric_info = report[selected_metric]

        scores = metric_info.get(
            "individual_scores", []
        )

        average_score = metric_info.get(
            "average_score", 0
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Average Score",
                f"{average_score * 100:.1f}%"
            )

        with col2:

            st.metric(
                "Evaluated Questions",
                len(scores)
            )

        st.divider()

        if scores:

            score_df = pd.DataFrame({
                "Question Number": range(1, len(scores) + 1),
                "Score (0-1)": scores,
                "Score (%)": [
                    float(score) * 100
                    for score in scores
                ]
            })

            fig = px.bar(
                score_df,
                x="Question Number",
                y="Score (%)",
                title=f"{selected_metric} - Individual Scores",
                range_y=[0, 100],
                color="Score (%)",
                color_continuous_scale=[
                    "#D8C8F5",
                    "#7653C6",
                    "#39205F"
                ]
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.subheader("Individual Question Scores")

            st.dataframe(
                score_df,
                width="stretch",
                hide_index=True
            )

        else:

            st.warning("No individual scores available.")

        st.divider()

        # Build complete CSV report
        csv_rows = []

        for metric_name, metric_data in report.items():

            for index, score in enumerate(
                metric_data.get("individual_scores", []),
                start=1
            ):

                csv_rows.append({
                    "Metric": metric_name,
                    "Question Number": index,
                    "Score (0-1)": score,
                    "Score (%)": float(score) * 100
                })

        csv_df = pd.DataFrame(csv_rows)

        st.download_button(
            "Download Complete CSV Report",
            csv_df.to_csv(index=False).encode("utf-8"),
            file_name="rag_evaluation_report.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download JSON Report",
            json.dumps(report, indent=4),
            file_name="rag_evaluation_report.json",
            mime="application/json"
        )

    else:

        st.warning(
            "Evaluation report not found. "
            "Run your evaluation first."
        )


# ==========================================
# UPLOAD PDF PAGE
# ==========================================

elif page == "Upload PDF":

    st.markdown(
        '<div class="main-title">'
        'Upload PDF Document</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload PDF documents to build your RAG knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            "**Selected documents:**",
            len(uploaded_files)
        )

        for file in uploaded_files:

            st.write("📄", file.name)

        if st.button("Process and Index PDFs"):

            from modules.document_processor import (
                extract_text_from_pdf,
                split_text_into_chunks
            )

            from modules.embeddings import (
                generate_embeddings
            )

            from modules.retriever import (
                store_embeddings
            )

            for uploaded_file in uploaded_files:

                try:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(
                                uploaded_file.getvalue()
                            )

                            temp_pdf_path = temp_file.name

                        try:

                            extracted_text = extract_text_from_pdf(
                                temp_pdf_path
                            )

                            if not extracted_text.strip():

                                st.warning(
                                    f"No readable text found in "
                                    f"{uploaded_file.name}"
                                )

                                continue

                            chunks = split_text_into_chunks(
                                extracted_text
                            )

                            if not chunks:

                                st.warning(
                                    f"No text chunks found in "
                                    f"{uploaded_file.name}"
                                )

                                continue

                            embeddings = generate_embeddings(
                                chunks
                            )

                            # Give the evaluation PDF a unique name
                            if uploaded_file.name.lower() == "sample.pdf":

                                document_name = (
                                    "AI_Evaluation_Document.pdf"
                                )

                            else:

                                document_name = uploaded_file.name

                            document_id = store_embeddings(
                                chunks,
                                embeddings,
                                document_name
                            )

                            st.success(
                                f"{uploaded_file.name} indexed successfully!"
                            )

                            st.write(
                                "**Stored document name:**",
                                document_name
                            )

                            st.write(
                                "**Document ID:**",
                                document_id
                            )

                            st.write(
                                "**Total chunks:**",
                                len(chunks)
                            )

                        finally:

                            if os.path.exists(temp_pdf_path):

                                os.remove(temp_pdf_path)

                except Exception as error:

                    st.error(
                        f"Failed to process {uploaded_file.name}: {error}"
                    )

            st.info(
                "PDF processing completed. "
                "You can now select documents in the next stage."
            )
