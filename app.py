import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")


def login():
    st.markdown("<h2 style='text-align:center;'> Login</h2>", unsafe_allow_html=True)

    users = {
        "admin": "1234",
        "user": "password"
    }

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()


st.markdown("""
<style>
.main {background-color: #0E1117;}

.chat-container {max-width: 850px; margin: auto;}

.user-msg {
    background: #DCF8C6;
    color: black;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    margin-left: auto;
    max-width: 75%;
}

.bot-msg {
    background: #2b2b2b;
    color: white;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 8px 0;
    margin-right: auto;
    max-width: 75%;
}

.header {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'> AI PDF Chatbot</div>", unsafe_allow_html=True)


def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d{8,}", "", text)
    text = re.sub(r"\s+", " ", text)
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
    outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_key" not in st.session_state:
    st.session_state.file_key = 0

st.sidebar.header(" Document")

st.sidebar.write(f" Logged in as: {st.session_state.username}")

if st.sidebar.button("➕ New Chat"):
    st.session_state.messages = []
    st.session_state.db = None
    st.session_state.file_key += 1
    st.rerun()

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.messages = []
    st.session_state.db = None
    st.rerun()

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type="pdf",
    key=st.session_state.file_key
)

if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        if not docs:
            st.error(" No readable text found in PDF")
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


question = st.chat_input("Ask a question about your document...")

if question:
    if st.session_state.db is None:
        st.warning(" Upload a PDF first")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        docs = st.session_state.db.similarity_search(question, k=3)

        if not docs:
            answer = "Not found in document"
        else:
            context = " ".join([clean_text(doc.page_content) for doc in docs])
            context = context[:2000]

            prompt = f"""
You are a helpful document assistant.

Answer ONLY using the context below.
If not found, say: "Not found in document".

Keep answer short and clear.

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