import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #DCF8C6;
    color: black;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: right;
    margin-left: auto;
    max-width: 75%;
}

.bot-msg {
    background-color: #2b2b2b;
    color: white;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: left;
    margin-right: auto;
    max-width: 75%;
}

.header {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'> AI PDF Chatbot</div>", unsafe_allow_html=True)


def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)  
    text = re.sub(r"\b\d{10,}\b", "", text)  
    text = re.sub(r"\n+", " ", text)
    return text.strip()


@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2  
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.header(" Upload Document")

if st.sidebar.button(" Clear Chat"):
    st.session_state.messages = []

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")


if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.error("No readable text found in PDF.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.db = FAISS.from_documents(texts, embeddings)

    st.sidebar.success(" Document ready!")

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


question = st.chat_input(" Ask a question about your document...")


if question:
    if st.session_state.db is None:
        st.warning("Upload a PDF first")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        docs = st.session_state.db.similarity_search(question, k=4)

        context = " ".join([clean_text(doc.page_content) for doc in docs])
        context = context[:1500]

        prompt = f"""


Rules:
- Answer ONLY from the context
- If not found, say: "Not found in document"
- Do NOT guess
- Keep answer short and accurate

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Thinking..."):
            answer = generate_answer(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.strip()
        })

        st.rerun()