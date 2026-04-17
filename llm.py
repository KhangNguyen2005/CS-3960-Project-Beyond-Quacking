import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

def summarize(text):
    prompt = f"Summarize this in one sentence:\n{text}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def answer(query, context):
    prompt = f"""
Answer the question using ONLY the context below.
Write 3-5 clear sentences.
If the context is not enough, say "Not enough information."

Context:
{context}

Question:
{query}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text