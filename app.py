import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Load API Key
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Page Setup
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚"
)

st.title("📚 RAG PDF Chatbot")
st.write("Upload your PDF and ask questions from it 🤖")


# PDF Text Extraction
def get_pdf_text(pdf):

    text = ""

    pdf_reader = PdfReader(pdf)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text



# Text Chunking
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks



# Create Vector Database
def create_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")



# Create Gemini Chain
def get_chain():

    prompt = """
    Answer the question using the given context.
    If the answer is not available in the context,
    say "Answer not available in the document".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )


    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["context","question"]
    )


    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt_template
    )

    return chain



# Ask Question
def ask_question(question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    database = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


    docs = database.similarity_search(question)


    chain = get_chain()


    response = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )


    st.write(response["output_text"])



# Sidebar
with st.sidebar:

    st.header("Upload PDF")

    pdf = st.file_uploader(
        "Choose PDF",
        type="pdf"
    )


    if st.button("Process PDF"):

        with st.spinner("Processing..."):

            text = get_pdf_text(pdf)

            chunks = get_text_chunks(text)

            create_vector_store(chunks)

            st.success("PDF Processed Successfully ✅")



question = st.text_input(
    "Ask a question from your PDF"
)


if question:

    ask_question(question)