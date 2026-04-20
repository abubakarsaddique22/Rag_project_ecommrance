# 🛒 E-Commerce RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** powered product assistant for e-commerce platforms. This project lets users ask natural language questions about products and get accurate, context-aware answers — built with LangChain, FastAPI, and vector store retrieval.

---

## 📌 Table of Contents

- [What is RAG?](#-what-is-rag)
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the App](#running-the-app)
- [API Endpoints](#-api-endpoints)
- [Modules Explained](#-modules-explained)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🤖 What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture that combines two powerful components:

1. **Retriever** — Searches a vector database to find the most relevant documents or product data for a user's query.
2. **Generator** — Passes the retrieved context along with the user's question to a Large Language Model (LLM), which then generates a precise, grounded answer.

Instead of the LLM relying only on its training data, RAG grounds answers in **your actual product data**, reducing hallucinations and improving accuracy.

```
User Query
    ↓
[Vector Store] → Retrieve Relevant Products/Docs
    ↓
[LLM (Gemini / Groq / Ollama)] → Generate Answer using Context
    ↓
Final Response to User
```

---

## 📋 Project Overview

This project is an **e-commerce product chatbot** that:
- Ingests product data (CSV/JSON) into a vector store
- Embeds product descriptions using Google Generative AI embeddings
- Stores vectors in **pinecone**
- Serves a **FastAPI** web application with a chat UI
- Uses **LangChain** to orchestrate the full RAG pipeline
- Supports multiple LLM backends: **Groq** and  **Ollama**

---

## ✨ Features

- 🔍 **Semantic Search** — Finds products based on meaning, not just keywords
- 💬 **Conversational Chat Interface** — Clean web UI built with HTML/CSS/JS and Jinja2 templates
- 🧠 **Multiple LLM Support** — Plug in Groq (Llama),local Ollama models
- 🗄️ **Flexible Vector Store** — Works with pinecone (cloud) or Chroma (local)
- ⚡ **Fast API Backend** — Built with FastAPI and Uvicorn for high performance
- 📦 **Modular Architecture** — Clean separation of concerns across ingestion, retrieval, prompts, and utilities
- 📓 **Jupyter Notebooks** — Exploratory notebooks for prototyping and testing

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI + Uvicorn |
| **LLM Orchestration** | LangChain |
| **LLM Providers** |Groq (Llama), Ollama |
| **Embeddings** | `langchain_ollama` |
| **Vector Store** | AstraDB (`langchain_pinecone`) / Chroma (`langchain-chroma`) |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 Templates) |
| **Data Processing** | Pandas |
| **Environment Config** | python-dotenv |
| **Package Manager** | `uv` (with `pyproject.toml`) |
| **Python Version** | 3.13+ |

---

## 📁 Project Structure

```
Rag_project_ecommrance/
│
├── main.py                    # FastAPI app entry point — routes & RAG chain
│
├── config/                    # Configuration files (API keys, settings)
│
├── data/                      # Raw product data (CSV/JSON files)
│
├── data_ingestion/            # Scripts to load, chunk & embed data into vector store
│
├── retriever/
│   └── retrieval.py           # Loads and returns the vector store retriever
│
├── utils/
│   └── model_loader.py        # Loads LLM and embedding models
│
├── prompt_library/
│   └── prompt.py              # Prompt templates (e.g., product_bot template)
│
├── templates/
│   └── chat.html              # Jinja2 chat UI template
│
├── static/                    # CSS, JS, and image assets for the frontend
│
├── notebook/                  # Jupyter notebooks for experimentation
│
├── pyproject.toml             # Project dependencies (uv/pip)
├── setup.py                   # Package setup
├── .python-version            # Python version pin (3.13)
├── .gitignore                 # Files excluded from git
└── uv.lock                    # Locked dependency versions
```

---

## ⚙️ How It Works

### Step 1 — Data Ingestion
Product data (e.g., a CSV with product names, descriptions, prices) is loaded using **Pandas**, split into chunks, embedded using **ollama AI embeddings**, and stored in a **vector database** (pinecone or Chroma).

### Step 2 — User Sends a Query
The user types a question in the chat UI, e.g., *"Do you have waterproof running shoes under $100?"*

### Step 3 — Retrieval
The `Retriever` module converts the query into an embedding and performs a **semantic similarity search** against the vector store to find the most relevant product entries.

### Step 4 — Prompt Construction
The retrieved product context is injected into a **LangChain prompt template** (`product_bot`) along with the original user question.

### Step 5 — LLM Generation
The constructed prompt is passed to the configured LLM (**Groq**), which generates a helpful, grounded response.

### Step 6 — Response Returned
The answer is sent back to the chat UI via the FastAPI `/get` endpoint and displayed to the user.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.13+**
- `uv` package manager (recommended) or `pip`
- An account on one of the following:
  - [Google AI Studio](https://aistudio.google.com/) for Gemini API key
  - [Groq Console](https://console.groq.com/) for Groq API key
  - [pinecone](https://app.pinecone.io/) for the vector database (or use local Chroma)

---

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/abubakarsaddique22/Rag_project_ecommrance.git
cd Rag_project_ecommrance
```

**2. Install dependencies using `uv` (recommended):**
```bash
pip install uv
uv sync
```

Or using standard pip:
```bash
pip install -r requirements.txt
```
> If there is no `requirements.txt`, you can generate one from `pyproject.toml`:
> ```bash
> pip install fastapi uvicorn langchain langchain-astradb langchain-chroma langchain-google-genai langchain-groq langchain-ollama pandas python-multipart jinja2 python-dotenv
> ```

---

### Environment Variables

Create a `.env` file in the root directory and add your credentials:

```env
# LLM Provider — choose one
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# AstraDB Vector Store (if using pinecone)
PINECONE_API_KEY = Your Api key of pinecone

# Optional: Ollama base URL (if running locally)
OLLAMA_BASE_URL=http://localhost:11434
```

> ⚠️ Never commit your `.env` file to GitHub. It is already listed in `.gitignore`.

---

### Data Ingestion

Before running the app, you need to ingest your product data into the vector store:

```bash
# Run the data ingestion script (check data_ingestion/ folder for the exact filename)
python data_ingestion/ingestion_pipeline.py
```

This will:
1. Load product data from the `data/` folder
2. Generate embeddings
3. Store them in the configured vector database

---

### Running the App

```bash
uvicorn main:app --reload
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

You will see the chat interface where you can start asking product-related questions.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the chat UI (HTML) |
| `POST` | `/get` | Accepts a user message, runs the RAG chain, returns the answer |

**Example POST request:**
```bash
curl -X POST "http://127.0.0.1:8000/get" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=Do you have wireless headphones?"
```

---

## 🧩 Modules Explained

### `main.py`
The application entry point. Sets up FastAPI, mounts static files and templates, initializes the `ModelLoader` and `Retriever`, and defines the RAG chain using LangChain's `RunnablePassthrough` pattern.

### `retriever/retrieval.py`
Handles loading the vector store and returning a LangChain-compatible retriever object used for similarity search.

### `utils/model_loader.py`
Centralizes loading of the LLM (Gemini / Groq / Ollama) and embedding model. This makes it easy to switch between providers without changing other parts of the code.

### `prompt_library/prompt.py`
Stores reusable prompt templates as a dictionary. The `product_bot` template instructs the LLM to answer only based on provided context, keeping responses accurate and on-topic.

### `data_ingestion/`
Contains scripts to read product data from the `data/` folder, process it with Pandas, embed it, and push it to the vector store.

### `config/`
Holds configuration constants like collection names, chunk sizes, and other settings that can be tuned without touching core logic.

---

## 🔮 Future Improvements

- [ ] Add conversation memory so the chatbot remembers previous messages in a session
- [ ] Add support for image-based product search
- [ ] Build a product filtering sidebar (price range, category)
- [ ] Add user authentication
- [ ] Deploy on AWS / GCP / Render with Docker
- [ ] Add evaluation metrics (RAGAS) to measure retrieval and generation quality
- [ ] Support streaming responses for a better UX

---

## 👨‍💻 Author

**Abubakar Saddique**
- GitHub: [@abubakarsaddique22](https://github.com/abubakarsaddique22)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please give it a star on GitHub!