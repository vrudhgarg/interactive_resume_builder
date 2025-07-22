# app.py

import streamlit as st
from qa_pipeline import get_qa_chain

# Set up the page
st.set_page_config(page_title="Lecture RAG", layout="wide")

st.title("📚 Ask your Lecture Notes")

# User input
query = st.text_input("Enter your question:")

# Run only if there's input
if query:
    with st.spinner("Searching your notes..."):
        qa_chain = get_qa_chain()
        response = qa_chain.invoke({"query": query})
        answer = response["result"]
        sources = response["source_documents"]

    st.markdown("### ✅ Answer:")
    st.success(answer)

    st.markdown("### 📂 Sources Used:")
    for doc in sources:
        source = doc.metadata.get("source", "Unknown")
        with st.expander(f"📄 {source}"):
            st.write(doc.page_content[:1000])  # You can limit content length
