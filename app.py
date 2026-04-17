import streamlit as st
import csv
import numpy as np
import os
from db import setup
from rag import embed, cosine
from llm import answer

sample_questions = [
    "What are applications of AI across industries?",
    "How is AI used in finance?",
    "What is AI used for in gaming?",
    "How do governments regulate AI?"
]

query = st.text_input(
    "Enter your question (or pick one below):",
    placeholder="e.g., How is AI used in finance?"
)

sample = st.selectbox("Or choose a sample:", [""] + sample_questions)

if not query:
    query = sample

st.title("DuckDB + Gemini RAG Demo")
st.write("This demo integrates DuckDB, embeddings, and Gemini for retrieval-augmented question answering (RAG).")

@st.cache_resource
def init_db():
    con = setup()

    con.execute("DELETE FROM documents")  # prevent duplicates

    file_path = os.path.join("data", "documents.csv")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        doc_id = int(row["id"])
        title = row["title"]
        content = row["content"]
        emb = embed(content).tolist()

        con.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?)",
            (doc_id, title, content, emb)
        )

    return con

def retrieve(query, con, k=3):
    q_emb = embed(query)
    docs = con.execute("SELECT title, content, embedding FROM documents").fetchall()

    scored = []
    for title, content, emb in docs:
        emb = np.array(emb)
        score = cosine(q_emb, emb)
        scored.append((score, title, content))

    scored.sort(reverse=True)
    return scored[:k]

con = init_db()


if st.button("Run Query") and query:
    top_docs = retrieve(query, con, k=3)

    st.subheader("Retrieved Documents")
    context_parts = []

    for score, title, content in top_docs:
        st.markdown(f"### {title}")
        st.write(f"Similarity: {score:.4f}")
        st.write(content)
        context_parts.append(content)

    context = "\n".join(context_parts)

    response = answer(query, context)

    st.subheader("Final Answer")
    st.write(response)