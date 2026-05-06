import streamlit as st
import sqlite3
import hashlib
import re
import os

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
    padding: 12px;
    border-radius: 15px;
    margin: 8px 0;
    margin-left: auto;
    max-width: 75%;
}

.bot-msg {
    background: #2b2b2b;
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 8px 0;
    margin-right: auto;
    max-width: 75%;
}

.header {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'> AI PDF Chatbot</div>", unsafe_allow_html=True)

DB_FILE = "users.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT,
            salt TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def valid_email(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)

def strong_password(password):
    return len(password) >= 6 and re.search(r"[A-Z]", password) and re.search(r"[0-9]", password)

def hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()

def register_user(email, password):
    if not valid_email(email):
        return False, "Invalid email"

    if not strong_password(password):
        return False, "Password must have capital letter & number"

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    salt = os.urandom(16).hex()
    hashed = hash_password(password, salt)

    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (email, hashed, salt))
        conn.commit()
        return True, "Registered successfully"
    except:
        return False, "User already exists"
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT password, salt FROM users WHERE email=?", (email,))
    user = c.fetchone()
    conn.close()

    if not user:
        return False

    stored_password, salt = user
    return stored_password == hash_password(password, salt)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d{8,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


if "user" not in st.session_state:
    st.session_state.user = None

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_key" not in st.session_state:
    st.session_state.file_key = 0

menu = st.sidebar.radio("Menu", ["Login", "Register"])

if menu == "Register":
    st.subheader("Create Account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not email or not password or not confirm:
            st.error("All fields required")
        elif password != confirm:
            st.error("Passwords do not match")
        else:
            success, msg = register_user(email, password)
            st.success(msg) if success else st.error(msg)

elif menu == "Login" and st.session_state.user is None:
    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(email, password):
            st.session_state.user = email
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")


if st.session_state.user:

    st.sidebar.success(f"{st.session_state.user}")

    if st.sidebar.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.db = None
        st.session_state.file_key += 1
        st.rerun()

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", key=st.session_state.file_key)


    if uploaded_file and st.session_state.db is None:
        with st.spinner("Processing document..."):

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            texts = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.db = FAISS.from_documents(texts, embeddings)

        st.sidebar.success("Document ready!")

  
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        style = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='{style}'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    question = st.chat_input("Ask a question about your document...")

    if question:
        if st.session_state.db is None:
            st.warning("Upload a PDF first")
        else:
            st.session_state.messages.append({"role": "user", "content": question})

       
            docs = st.session_state.db.similarity_search(question, k=6)

            context = " ".join([clean_text(doc.page_content) for doc in docs])

            if question.lower() not in context.lower():
                matched = []
                for doc in docs:
                    if any(word in doc.page_content.lower() for word in question.lower().split()):
                        matched.append(doc.page_content)
                if matched:
                    context = " ".join(matched)

            context = context[:3000]

            if not context.strip():
                answer = "Not found in document"
            else:
                prompt = f"""
Answer the question using ONLY the context.
Be flexible with wording.

Context:
{context}

Question:
{question}

Answer:
"""

                with st.spinner("Thinking..."):
                    answer = generate_answer(prompt)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

else:
    st.warning("Please login to use the chatbot")