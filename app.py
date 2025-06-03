import pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import gradio as gr


df = pd.read_csv("knowledge_base.csv")


embedder = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding"] = embedder.encode(df["question"].tolist(), convert_to_tensor=True).tolist()

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

def retrieve_context(query):
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, df["embedding"].tolist())[0]
    best_idx = similarities.argmax().item()
    return df.iloc[best_idx]["answer"]

def generate_response(user_input):
    context = retrieve_context(user_input)
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nBot:"
    result = client.text_generation(prompt, max_new_tokens=100)
    return result

iface = gr.Interface(fn=generate_response,
                     inputs=gr.Textbox(label="Ask something"),
                     outputs=gr.Textbox(label="Bot Response"),
                     title="RAG Chatbot",
                     description="Chatbot with RAG over your CSV knowledge base")

if __name__ == "__main__":
    iface.launch()