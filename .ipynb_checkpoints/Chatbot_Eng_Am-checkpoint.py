import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langdetect import detect
import os

# ---- Setup ----
os.environ["TRANSFORMERS_NO_TF"] = "1"
FAISS_DB_PATH = "faiss_faqs_db1"  # Adjust if needed

# Load multilingual embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Load LLM via Ollama (change model name if needed)
llm = Ollama(model="deepseek-llm")

# Load FAISS vector store
if os.path.exists(FAISS_DB_PATH):
    vector_store = FAISS.load_local(
        FAISS_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
else:
    st.error("❌ FAISS DB not found. Please process documents first.")
    st.stop()

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ---- Streamlit UI ----
st.set_page_config(page_title="CBE Chatbot", page_icon="💬")
st.title("💬 CBE AI Chatbot")
st.caption("Ask your question in English or Amharic.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# New user input
user_message = st.chat_input("Ask in Amharic or English...")

if user_message:
    st.chat_message("user").markdown(user_message)

    try:
        # Detect input language
        lang = detect(user_message)
        # Optional: Debug detected language
        # st.info(f"Detected language: {lang}")

        # Language-specific prompt
        if lang == "am":
            prompt = (
                "አንተ የኮሜርሻል ባንክ ኢትዮጵያ ላማ 2 በሚባል ሞዴል የተሰራ አስተማማኝ መረጃ ረዳት ነህ። "
                "ጥያቄዎችን በአማርኛ በግልጽነትና በትክክል መልስ ስጥ።\n\n"
                f"ተጠያቂ፡ {user_message.strip()}\nመልስ፡"
            )
        else:
            prompt = (
                "You are an AI assistant trained on internal Commercial Bank of Ethiopia documents using Llama2. "
                "Respond to questions clearly in English.\n\n"
                f"User: {user_message.strip()}\nAssistant:"
            )

        # Validate user message content
        if user_message.strip() and len(user_message.strip()) > 2:
            response = qa_chain.run(prompt)

            if not response or response.strip() == "":
                response = (
                    "ይቅርታ፣ ጥያቄዎትን አልተረዳሁም። እባክዎ እንደገና ይሞክሩ።" if lang == "am"
                    else "Sorry, I couldn't find a clear answer. Please try rephrasing your question."
                )
        else:
            response = (
                "እባክዎ ትክክለኛ ጥያቄ ያስገቡ።" if lang == "am"
                else "Please enter a valid question."
            )

        # Show assistant's response
        with st.chat_message("assistant"):
            st.markdown(f"💡 **Answer:** {response}")

        # Save chat history
        st.session_state.chat_history.append({
            "user": user_message.strip(),
            "bot": f"💡 **Answer:** {response}"
        })

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
