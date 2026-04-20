import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")


st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: right;
}

.bot-msg {
    background-color: #2b2b2b;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.title(" AI PDF Chatbot")


@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(texts, embeddings)
        st.session_state.db = db

    st.sidebar.success(" Document ready!")


st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-msg'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-msg'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

question = st.chat_input("Ask a question about your document...")

if question:
    if not st.session_state.db:
        st.warning("Please upload a PDF first")
    else:
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Retrieve relevant chunks
        docs = st.session_state.db.similarity_search(question, k=4)
        context = " ".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        with st.spinner("Thinking..."):
            answer = generate_answer(prompt)

        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()