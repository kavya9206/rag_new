import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

# ------------ CONFIG -----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📘 RAG Document Q&A System (Streamlit)")
st.write("Upload documents, ask questions, and get answers grounded in your files.")

# --------- Load Embedding Model ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

embedder = load_embedder()

# --------- Document Uploader Section ----------
uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    documents = []

    for file in uploaded_files:
        file_path = f"temp/{file.name}"
        os.makedirs("temp", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(file.read())

        # Load PDFs or TXT
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents.extend(loader.load())

    # --------- Chunking Text ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    docs = splitter.split_documents(documents)

    st.success(f"Documents loaded & split into {len(docs)} chunks!")

    # --------- Create Vectorstore (Chroma) ----------
    vectordb = Chroma.from_documents(
        docs,
        embedder,
        persist_directory="vector_db"
    )

    st.success("Vector database created!")

    # ----------- Q&A Interface -------------
    user_question = st.text_input("Ask a question about your documents:")

    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            # Retrieve Relevant Chunks
            matched_docs = vectordb.similarity_search(user_question, k=4)

            context = "\n\n".join([d.page_content for d in matched_docs])

            # Build Prompt
            prompt = f"""
You are an assistant. Answer ONLY using the information in the context. 
If you don't know the answer, say "I don't know."

Context:
{context}

Question: {user_question}

Answer:
"""

            # Call OpenAI LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

            st.subheader("📌 Answer")
            st.write(answer)

            with st.expander("🔍 Retrieved Document Chunks"):
                for i, doc in enumerate(matched_docs):
                    st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")
else:
    st.info("Upload documents to begin.")