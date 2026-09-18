import os
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Chatbot")
st.write(
    "Upload multiple PDF documents and ask questions from them using RAG 🤖"
)


# =========================================================
# 3. PDF TEXT EXTRACTION
# =========================================================

def get_pdf_documents(pdf_files):

    documents = []

    for pdf in pdf_files:

        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(pdf_reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                documents.append({
                    "text": text,
                    "source": pdf.name,
                    "page": page_number
                })

    return documents


# =========================================================
# 4. TEXT CHUNKING
# =========================================================

def get_text_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(document["text"])

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document["page"]
            })

    return chunks


# =========================================================
# 5. CREATE FAISS VECTOR DATABASE
# =========================================================

def create_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
       model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    texts = []
    metadatas = []

    for chunk in chunks:

        texts.append(chunk["text"])

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"]
        })

    vector_store = FAISS.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vector_store.save_local("faiss_index")

    return vector_store


# =========================================================
# 6. GEMINI MODEL
# =========================================================

def get_llm():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    return model

# =========================================================
# 7. PROMPT
# =========================================================

def get_prompt():

    prompt = """
You are a document-based AI assistant.

You have access to information retrieved from multiple PDF documents.

Answer the user's question using ONLY the information provided
in the context.

IMPORTANT RULES:

1. The context may contain information from different PDF files.
2. First identify which information belongs to the person/resume.
3. Then identify which information belongs to the internship/task document.
4. If the question asks whether a person's skills match the
   requirements of a task, compare BOTH sets of information.
5. Clearly mention:
   - Skills the person possesses
   - Skills/requirements needed for the task
   - Which requirements match
   - Which requirements are not found in the person's document
6. Do not assume that a skill exists unless it is explicitly
   present in the uploaded documents.
7. Do not use outside knowledge.
8. If the documents do not contain enough information to make
   the comparison, say:
   "Answer not available in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        template=prompt,
        input_variables=["context", "question"]
    )


# =========================================================
# 8. ASK QUESTION
# =========================================================

# =========================================================
# 8. ASK QUESTION
# =========================================================

def ask_question(question):

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Load FAISS vector database
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=10
    )

    # If no relevant documents found
    if not docs:
        return (
            "Answer not available in the uploaded documents.",
            []
        )

    # Combine retrieved documents
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )

    # Create prompt
    prompt = get_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Gemini
    llm = get_llm()

    response = llm.invoke(final_prompt)

    # Convert Gemini response into clean text
    if isinstance(response.content, str):

        answer = response.content

    else:

        answer = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    # Remove duplicate sources
    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        source_info = f"{source} — Page {page}"

        if source_info not in sources:
            sources.append(source_info)

    return answer, sources

# =========================================================
# 9. SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# 10. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Upload Documents")

    pdf_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    process_button = st.button(
        "🚀 Process PDFs",
        use_container_width=True
    )

    if process_button:

        if not pdf_files:

            st.warning("Please upload at least one PDF.")

        else:

            with st.spinner("Processing PDFs..."):

                try:

                    # Extract text
                    documents = get_pdf_documents(pdf_files)

                    if not documents:

                        st.error(
                            "No readable text found in the uploaded PDFs."
                        )

                    else:

                        # Chunk documents
                        chunks = get_text_chunks(documents)

                        # Create vector database
                        create_vector_store(chunks)

                        # Clear previous chat
                        st.session_state.messages = []

                        st.success(
                            f"Successfully processed "
                            f"{len(pdf_files)} PDF(s)!"
                        )

                        st.info(
                            f"Created {len(chunks)} text chunks."
                        )

                except Exception as e:

                    st.error(
                        f"Error while processing PDFs: {str(e)}"
                    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 11. DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown("**📚 Sources:**")

            for source in message["sources"]:

                st.caption(f"📄 {source}")


# =========================================================
# 12. USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask a question from your uploaded PDFs..."
)


if question:

    # Check whether vector database exists
    if not os.path.exists("faiss_index"):

        st.warning(
            "Please upload and process your PDFs first."
        )

    else:

        # Display user question
        with st.chat_message("user"):

            st.markdown(question)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                try:

                    answer, sources = ask_question(question)

                    st.markdown(answer)

                    # Display sources
                    if sources:

                        st.markdown("**📚 Sources:**")

                        for source in sources:

                            st.caption(f"📄 {source}")

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    error_message = f"Error: {str(e)}"

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })