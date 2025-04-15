# 🧠 Content-Based ChatBot

This is an interactive content-based chatbot built using **LangChain**, **Ollama (LLaMA 3)**, **FAISS** for vector storage, and **PostgreSQL** for chat history logging. It fetches content from websites, intelligently answers user questions based on that content, and displays recent conversations.

---

## 🚀 Features

- 💬 Ask context-based questions about web content (e.g., LangChain documentation)
- ⚡ Powered by **LLaMA 3** via **Ollama**
- 📄 Web content loaded and split into vector embeddings using **FAISS**
- 🧠 Uses **LangChain's Retrieval-Augmented Generation (RAG)** pipeline
- 💾 Stores chat history (user & bot messages) in PostgreSQL
- 📜 Displays recent chat history with a single checkbox
- 🖥️ Interactive frontend built using **Streamlit**

---

## 📦 Tech Stack

- **Frontend**: Streamlit
- **LLM**: LLaMA3 via Ollama
- **Embeddings**: `nomic-embed-text`
- **Web Loader**: LangChain's `WebBaseLoader`
- **Vector DB**: FAISS
- **Database**: PostgreSQL (with `psycopg2`)
- **RAG**: LangChain `retrieval_chain`

---

## 🧠 How It Works

1. Loads content from [LangChain documentation](https://docs.smith.langchain.com/)
2. Splits the content into chunks and converts them into embeddings
3. Stores those embeddings in a FAISS vector store
4. Accepts user input via Streamlit
5. Finds relevant chunks using vector similarity
6. Sends both the context and question to **LLaMA 3** via Ollama
7. Saves the question and answer to PostgreSQL
8. Displays the most recent 5 conversations on demand

---

## 📄 PostgreSQL Schema

Make sure you create the following table:

```sql
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_msg TEXT NOT NULL,
    bot_msg TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

✨ Usage
Type a question related to the loaded web content (LangChain docs).
The chatbot answers based on document similarity.
Enable "Show previous Chats" to view the last 5 interactions.

✍️ Author
Made by Jayani
