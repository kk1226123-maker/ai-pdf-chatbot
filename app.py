import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page config
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title(" AI PDF Chatbot")

# Session state
if "db" not in st.session_state:
    st.session_state.db = None

if "history" not in st.session_state:
    st.session_state.history = []

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        st.session_state.db = db

    st.success("Document ready!")

# Ask question
question = st.text_input("Ask a question about the document")

def extract_answer(question, docs):
    question_words = question.lower().split()
    best_chunks = []

    for doc in docs:
        text = doc.page_content.lower()
        score = sum(word in text for word in question_words)

        if score > 0:
            best_chunks.append((score, doc.page_content))

    if best_chunks:
        best_chunks.sort(reverse=True)
        return "\n\n".join([chunk for _, chunk in best_chunks[:2]])

    return docs[0].page_content

if question:
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(question, k=4)
        answer = extract_answer(question, docs)

        st.session_state.history.append((question, answer[:700]))
    else:
        st.warning(" Please upload a PDF first")

# Chat history
if st.session_state.history:
    st.subheader(" Chat History")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")