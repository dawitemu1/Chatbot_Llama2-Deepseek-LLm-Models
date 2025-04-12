# 💬 Financial Sector Chatbot with LLaMA 2 & DeepSeek

A Retrieval-Augmented Generation (RAG) chatbot designed for financial sector Q&A, leveraging **LLaMA 2**, **DeepSeek**, and advanced NLP pipelines. Built for accuracy, security, and scalability.

![Demo](https://img.shields.io/badge/Demo-Streamlit-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🛠️ Technologies Used

| Component          | Technology Stack                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| **LLM**            | LLaMA 2 (via Ollama/Hugging Face) + DeepSeek                                    |
| **Embeddings**     | `sentence-transformers/all-MiniLM-L6-v2`                                        |
| **Vector Store**   | FAISS (local) / Pinecone (cloud)                                               |
| **RAG Framework**  | LangChain                                                                       |
| **Backend**        | FastAPI (REST) / Flask (WSGI)                                                  |
| **Frontend**       | Streamlit (prototyping) / React (production)                                   |
| **Deployment**     | Docker + AWS/GCP (optional)                                                    |

## ✨ Key Features

- **Financial-Specific RAG**: Pre-trained on SEC filings, earnings reports, and financial news.
- **Multi-Model Support**: Switch between LLaMA 2 and DeepSeek dynamically.
- **Secure**: API key management via `.env` + GitHub secret scanning.
- **Low Latency**: FAISS for local dev, Pinecone for cloud scaling.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (for LLaMA 2 local inference) / Hugging Face `transformers`
- [Pinecone API key](https://www.pinecone.io/) (optional)

### Installation
```bash
git clone https://github.com/your-repo/financial-chatbot.git
cd financial-chatbot
pip install -r requirements.txt
