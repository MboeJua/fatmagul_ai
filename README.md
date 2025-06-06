## RAG WhatsApp Chatbot with Google Cloud Run

A Retrieval-Augmented Generation (RAG) chatbot that integrates with WhatsApp via Twilio, using a Hugging Face model and a local CSV knowledge base.

---

## ✅ Features

- WhatsApp integration via Twilio Sandbox
- RAG-based contextual answers from `knowledge_base.csv`
- Lightweight model (`flan-t5-small`) for fast inference
- Deployable on Google Cloud Run (For local testing Use Ngrok)
- Grabs the most relevant answer using sentence similarity with `MiniLM-L6`

---

## 📁 Project Structure
├── app.py 
├── requirements.txt 
├── knowledge_base.csv # Your Q&A knowledge base
├── Dockerfile # Container build file
└── README.md


## Personal Setup

### 1. Clone the repo & add your CSV
```bash
git clone https://github.com/MboeJua/fatmagul_ai.git
cd fatmagul_ai
```
### 2. Run locally - Assuming ngrok and twilio setup
```
pip install -r requirements.txt
python app.py
```

