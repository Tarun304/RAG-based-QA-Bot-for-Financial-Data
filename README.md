# RAG-based-QA-Bot-for-Financial-Data
Detailed Documentation for the RAG-based QA Bot for Financial Data

Introduction:
This project implements a Retrieval-Augmented Generation (RAG) model for a Question Answering (QA) bot that processes Profit and Loss (P&L) statements from PDF documents. It combines retrieval-based techniques for fetching relevant financial data with generative techniques for answering user queries.
The project consists of two main components:
1.	Part 1: Backend Pipeline for Data Extraction, Embedding, and Retrieval
2.	Part 2: Interactive User Interface for Real-time Query Resolution

System Architecture:
1. Backend Pipeline
The backend pipeline processes financial documents and prepares data for efficient query handling:
1.	Data Extraction:
o	The unstructured library extracts tables and text from P&L statements in PDFs.
o	Extracted tables are converted into structured formats like Pandas DataFrames for further processing.
2.	Data Cleaning:
o	Empty rows and columns are dropped from the extracted tables.
o	Data is cleaned to ensure consistent formatting.
3.	Data Summarization:
o	Google Gemini AI is used to generate summaries of the extracted tables, providing concise descriptions of financial metrics.
4.	Vector Embeddings:
o	Sentence embeddings for tables and summaries are generated using the SentenceTransformer model (all-mpnet-base-v2).
o	These embeddings are stored in Pinecone, a vector database, for efficient retrieval.
5.	Query Processing:
o	For each user query, relevant contexts (e.g., summaries and table rows) are retrieved from Pinecone.
o	Google Gemini AI generates a final answer based on the retrieved data and user query.

2. Interactive User Interface
The frontend, built using Streamlit, provides users with the ability to:
1.	Upload P&L statements in PDF format.
2.	Ask financial questions related to the uploaded data.
3.	View precise answers alongside the relevant financial data.
The interface uses Streamlit's session state to manage uploaded data, processed tables, and user interaction
Requirements
Python Libraries
A requirements.txt file is provided to manage the dependencies:
System Setup
1.	Install required libraries:
2.	pip install -r requirements.txt
3.	Install additional system dependencies for PDF processing:
4.	sudo apt-get update
5.	sudo apt-get install -y poppler-utils libleptonica-dev tesseract-ocr libtesseract-dev
6.	Use the streamlit and localtunnel for serving the app:
7.	streamlit run app.py & npx localtunnel --port 8501

Step-by-Step Explanation
Part 1: Backend Pipeline
1. Data Extraction
The unstructured library extracts tables and elements from the PDF using the following:
raw_pdf_elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    hi_res_model_name="yolox",
    extract_images_in_pdf=True,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_output_dir="extracted_data",
)
2. Table Extraction
The extract_tables function isolates tables from the PDF elements:
tables = [el for el in elements if el.category == "Table"]
3. Cleaning Tables
The clean_table function ensures consistency in the extracted data:
if table is not None:
    table.dropna(how="all", axis=0, inplace=True)
    table.dropna(how="all", axis=1, inplace=True)
    table.reset_index(drop=True, inplace=True)
4. Summarization
Tables are summarized using Google Gemini AI, generating insights such as profit margins or revenue trends:
prompt = f"Summarize key insights from this financial table:\n\n{table_html}"
response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
5. Vector Embedding and Pinecone Upsert
Vector embeddings of tables and summaries are created using SentenceTransformer:
summary_embedding = embedding_model.encode(summary)
table_embedding = embedding_model.encode(table_html)
Data is then upserted to Pinecone for retrieval:
index.upsert(vectors=vectors)

Part 2: Interactive User Interface
1. File Upload
The user uploads a PDF file using Streamlit:
uploaded_file = st.file_uploader("Upload your financial PDF document", type=["pdf"])
2. Processing and Embedding
Uploaded files are processed, and tables are summarized and embedded. The processing status is displayed using a spinner:
with st.spinner("Processing the uploaded PDF..."):
    raw_pdf_elements = partition_pdf(...)
3. Query Processing
User queries are processed in real time. Relevant contexts are retrieved from Pinecone:
contexts = get_relevant_contexts(user_query, index)
Answers are generated using Google Gemini AI:
answer = generate_answer_with_gemini(contexts, user_query)
4. Interactive Loop
Users can ask questions repeatedly, and a chat history is maintained:
if user_query:
    st.session_state.qa_history.append({"question": user_query, "answer": answer})

Challenges and Solutions:
1.	Handling Complex PDFs:
o	PDFs with inconsistent table structures required fallback mechanisms (e.g., text-based parsing).
2.	Performance Optimization:
o	Pinecone ensures fast vector-based retrieval.
o	Streamlit's session state minimizes redundant computations.
3.	Accurate Summarization:
o	Google Gemini AI provides precise insights but was fine-tuned for financial contexts.

Sample Queries and Responses

Query: "What is the gross profit for Q3 2024??"
Response: The gross profit for Q3 2024 is 11,175.


Query: "What are the total liabilities as of March 31, 2024?"
Response: As of March 31, 2024, the company's contingent liabilities were ₹3,583 crore.

Deployment Instructions
1.	Clone the repository:
2.	git clone <repo_url>
3.	cd <repo_name>
4.	Install dependencies:
5.	pip install -r requirements.txt
6.	sudo apt-get install -y poppler-utils libleptonica-dev tesseract-ocr libtesseract-dev
7.	Run the application:
8.	streamlit run app.py
9.	Access via localtunnel:
10.	npx localtunnel --port 8501

Conclusion:
This RAG-based QA Bot is a robust solution for extracting and querying financial data from P&L statements. It effectively combines document parsing, vector-based retrieval, and generative AI, delivering precise answers and contextual references.


Submitted By:
Tarun Kumar Behera
Email: tkbehera.work304@gmail.com


