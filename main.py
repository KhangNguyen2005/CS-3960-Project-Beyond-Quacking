from db import setup
from rag import embed, cosine
from llm import answer
import csv
import numpy as np


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

print("Starting...")
con = setup()

print("Loading dataset...")

with open("data/documents.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Generating embeddings for documents...")

for row in rows:
    doc_id = int(row["id"])
    title = row["title"]
    content = row["content"]

    emb = embed(content).tolist()

    con.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?)",
        (doc_id, title, content, emb)
    )

query = "What are applications of AI across industries?"
print("Retrieving documents...")
top_docs = retrieve(query, con, k=3)

print("Retrieved context:")
context_parts = []

for score, title, content in top_docs:
    print(f"{score:.4f} | {title}")
    print(f"  {content}")
    context_parts.append(content)

context = "\n".join(context_parts)

print("Calling Gemini...")
response = answer(query, context)

print("\nFinal Answer:")
print(response)
print("Done.")