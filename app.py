import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title(" AI PDF Chatbot")

#  Load model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Session
if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,   
            chunk_overlap=80
        )

        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.db = FAISS.from_documents(texts, embeddings)

    st.sidebar.success(" Document ready!")


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f" **You:** {msg['content']}")
    else:
        st.markdown(f" **AI:** {msg['content']}")



question = st.chat_input("Ask a question about your document...")

if question:
    if st.session_state.db is None:
        st.warning("Upload a PDF first")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        
        docs = st.session_state.db.similarity_search(question, k=2)

        context = " ".join([doc.page_content for doc in docs])
        context = context[:1000]

        
        prompt = f"""
You are a document assistant.

ONLY answer from the provided context.
If the answer is not clearly in the context, say: "Not found in document".

Keeping  the answer short and exact.

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