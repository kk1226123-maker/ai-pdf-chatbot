import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

st.set_page_config(page_title="AI PDF Chatbot")

st.title("AI PDF Chatbot (Free + Local AI)")
st.write("Upload a PDF and ask questions")

# Load AI model
llm = Ollama(model="llama3")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="llama3")

    # Create vector database
    db = FAISS.from_documents(texts, embeddings)

    st.success("PDF processed successfully!")

    # Ask question
    query = st.text_input("Ask a question about your PDF")

    if query:
        docs = db.similarity_search(query)

        # Combine context
        context = " ".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {query}
        """

        answer = llm.invoke(prompt)

        st.write(" Answer:")
        st.write(answer)