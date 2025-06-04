from flask import Flask, request
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd

df = pd.read_csv("knowledge_base.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding"] = embedder.encode(df["question"].tolist(), convert_to_tensor=True).tolist()

generator = pipeline("text2text-generation", model="google/flan-t5-small")

def rag_answer(user_input):
    query_embedding = embedder.encode([user_input], convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, df["embedding"].tolist())[0]
    best_idx = similarities.argmax().item()
    context = df.iloc[best_idx]["answer"]
    prompt = f"Answer the question using the context.\nContext: {context}\nQuestion: {user_input}"
    return generator(prompt, max_new_tokens=200)[0]["generated_text"]

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    user_input = request.values.get("Body", "").strip()
    reply = rag_answer(user_input)
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

@app.route("/")
def home():
    return "RAG WhatsApp bot is running on Railway."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

