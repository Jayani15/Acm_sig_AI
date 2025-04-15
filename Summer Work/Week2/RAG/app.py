import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import psycopg2

from dotenv import load_dotenv
load_dotenv()

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="chatbot_db",
        user="postgres",
        password="Jayani15",
        port=5432
    )

def save_chat(user_msg, bot_msg):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (user_msg, bot_msg) VALUES (%s,%s)",(user_msg, bot_msg)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Failed to save chat: {e}")


if "vector" not in st.session_state:
    #Loading the vector store
    st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    #Splitting the documents into smaller chunks(Transforming)
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("Content based ChatBot")
llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("Input your prompt")

if user_question:
    response = retrieval_chain.invoke({"input":user_question})
    bot_answer = response['answer']

    st.write(response['answer'])

    save_chat(user_question, bot_answer)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")

def get_chat_history():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("sELECT user_msg, bot_msg, timestamp FROM chat_history ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Could not fetch history: {e}")
        return []
    
if st.checkbox("Show previous Chats"):
    for user_msg, bot_msg, timestamp in get_chat_history():
        st.markdown(f"** User Question ({timestamp}):** {user_msg}")
        st.markdown(f"**Bot Answer: ** {bot_msg}")