import streamlit as st
import json
import hashlib
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="AI PDF Chatbot", layout="wide")


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


USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password):
    users = load_users()
    if email in users:
        return False, "User already exists"
    users[email] = hash_password(password)
    save_users(users)
    return True, "Registered successfully"

def login_user(email, password):
    users = load_users()
    if email not in users:
        return False
    return users[email] == hash_password(password)


if "user" not in st.session_state:
    st.session_state.user = None

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_key" not in st.session_state:
    st.session_state.file_key = 0


menu = st.sidebar.selectbox("Menu", ["Login", "Register"])


if menu == "Register":
    st.subheader("Create Account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        success, msg = register_user(email, password)
        if success:
            st.success(msg)
        else:
            st.error(msg)


elif menu == "Login" and st.session_state.user is None:
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(email, password):
            st.session_state.user = email
            st.success("Logged in successfully")
            st.rerun()
        else:
            st.error("Invalid credentials")


if st.session_state.user:

    st.sidebar.success(f"Logged in as {st.session_state.user}")

    if st.sidebar.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.db = None
        st.session_state.file_key += 1
        st.rerun()

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type="pdf",
        key=st.session_state.file_key
    )

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

    def clean_text(text):
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d{8,}", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    if uploaded_file and st.session_state.db is None:
        with st.spinner("Processing document..."):

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            if not docs:
                st.error("No readable text found")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )

            texts = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.db = FAISS.from_documents(texts, embeddings)

        st.sidebar.success("Document ready!")


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
            st.warning("Upload a PDF first")
        else:
            st.session_state.messages.append({"role": "user", "content": question})

            results = st.session_state.db.similarity_search_with_score(question, k=5)
            docs = [doc for doc, score in results if score < 0.7]

            if not docs:
                answer = "Not found in document"
            else:
                context = " ".join([clean_text(doc.page_content) for doc in docs])
                context = context[:2000]

                prompt = f"""
You are a highly accurate document assistant.

Rules:
- Answer ONLY from context
- If not found say: Not found in document
- Keep answer short

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


else:
    st.warning("Please login to use the chatbot")