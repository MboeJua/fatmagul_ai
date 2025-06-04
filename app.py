from flask import Flask, request
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from twilio.twiml.messaging_response import MessagingResponse
import os

# Load data
kb = pd.read_csv("knowledge_base.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kb["embedding"] = embedder.encode(kb["question"].tolist(), convert_to_tensor=True).tolist()

generator = pipeline("text2text-generation", model="google/flan-t5-small")

# RAG answer function
def rag_answer(user_input):
    query_embedding = embedder.encode([user_input], convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb["embedding"].tolist())[0]
    best_idx = similarities.argmax().item()
    context = kb.iloc[best_idx]["answer"]
    prompt = f"Answer the question using the context.\nContext: {context}\nQuestion: {user_input}"
    return generator(prompt, max_new_tokens=100)[0]["generated_text"]

# Flask app
app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").strip()
    reply = rag_answer(incoming_msg)
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

@app.route("/")
def home():
    return "RAG WhatsApp Bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

