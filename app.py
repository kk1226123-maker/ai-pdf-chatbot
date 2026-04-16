import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Document Intelligence Tool", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: white;
    text-align: center;
}
p {
    text-align: center;
    color: #A0A0A0;
}
.question {
    font-weight: bold;
    margin-top: 15px;
}
.answer {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Document Intelligence Tool</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a document and extract insights instantly</p>", unsafe_allow_html=True)

if "db" not in st.session_state:
    st.session_state.db = None

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader(" Upload Document")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    with st.spinner("Analyzing document..."):
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

    st.success("Document processed successfully")

st.subheader(" Ask a Question")

question = st.text_input("Enter your question")

if question:
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(question, k=4)
        context = " ".join([doc.page_content for doc in docs])

        answer = f"Answer based on document:\\n\\n{context[:500]}..."

        st.session_state.history.append({
            "question": question,
            "answer": answer
        })

    else:
        st.warning("Please upload a document first")

if st.session_state.history:
    st.subheader(" Previous Questions")

    for item in reversed(st.session_state.history):
        st.markdown(f"<div class='question'>Q: {item['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer'>A: {item['answer']}</div>", unsafe_allow_html=True)