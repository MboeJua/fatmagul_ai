from transformers import pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Load knowledge base
df = pd.read_csv("knowledge_base.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding"] = embedder.encode(df["question"].tolist(), convert_to_tensor=True).tolist()

# Load T5-based model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def retrieve_context(query):
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, df["embedding"].tolist())[0]
    best_idx = similarities.argmax().item()
    return df.iloc[best_idx]["answer"]

def generate_response(user_input):
    context = retrieve_context(user_input)
    prompt = f"Answer the question using the context.\nContext: {context}\nQuestion: {user_input}"
    result = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    return result

iface = gr.Interface(fn=generate_response,
                     inputs=gr.Textbox(label="Ask something"),
                     outputs=gr.Textbox(label="Bot Response"),
                     title="RAG Chatbot (Flan-T5 Small)",
                     description="Fast RAG chatbot using Google Flan-T5-Small (local, no token needed)")

if __name__ == "__main__":
    iface.launch()
