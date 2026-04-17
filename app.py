import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Page config
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

# UI Styling
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    text-align: center;
    color: white;
}
.chat-box {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user {
    color: #4CAF50;
}
.bot {
    color: #00BFFF;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> AI PDF Chatbot</h1>", unsafe_allow_html=True)

#  Load better QA model
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

qa_pipeline = load_model()

# Session state
if "db" not in st.session_state:
    st.session_state.db = None

if "history" not in st.session_state:
    st.session_state.history = []

# Upload PDF
st.subheader("Upload your PDF")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(texts, embeddings)
        st.session_state.db = db

    st.success("Document ready!")

# Ask question
st.subheader("Ask Questions")
question = st.text_input("Type your question...")

if question:
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(question, k=4)
        context = " ".join([doc.page_content for doc in docs])

        with st.spinner("Thinking..."):
            prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""
            response = qa_pipeline(prompt, max_length=200)
            answer = response[0]["generated_text"].strip()

        st.session_state.history.append((question, answer))
    else:
        st.warning("Please upload a PDF first")

# Chat history
if st.session_state.history:
    st.subheader("Chat History")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"<div class='chat-box'><span class='user'>You:</span> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-box'><span class='bot'>AI:</span> {a}</div>", unsafe_allow_html=True)