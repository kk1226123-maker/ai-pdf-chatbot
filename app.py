import streamlit as st

st.title(" AI PDF Chatbot")
st.write("Upload your PDF and ask questions")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Ask question
user_question = st.text_input("Ask a question about your PDF")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

if user_question:
    st.write(" Answer:")
    st.write("This is where AI response will come")

