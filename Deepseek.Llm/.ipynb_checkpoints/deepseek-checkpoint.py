import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from googletrans import Translator
import os

# --------------------- Configuration ---------------------
os.environ["TRANSFORMERS_NO_TF"] = "1"
FAISS_DB_PATH = "faiss_faqs_db1"
translator = Translator()

# --------------------- Load LLM & Vector Store ---------------------
llm = Ollama(model="llama2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(FAISS_DB_PATH):
    try:
        vector_store = FAISS.load_local(
            FAISS_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.error(f"❌ Error loading FAISS vector store: {e}")
        st.stop()
else:
    st.error("❌ FAISS database not found. Please make sure the vector store is prepared.")
    st.stop()

# --------------------- QA Chain ---------------------
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --------------------- Streamlit Chatbot UI ---------------------
st.set_page_config(page_title="CBE Chatbot", page_icon="💬")
st.title("💬 CBE AI Chatbot")
st.caption("Multilingual Chatbot (English & Amharic) for Commercial Bank of Ethiopia.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# Chat input
user_message = st.chat_input("Ask in English or Amharic...")

if user_message:
    st.chat_message("user").markdown(user_message)

    try:
        # Detect language
        detected_lang = translator.detect(user_message).lang

        # Translate to English if it's Amharic or another language
        if detected_lang != "en":
            translated_input = translator.translate(user_message, dest='en').text
        else:
            translated_input = user_message

        # Prepare prompt and get answer
        prompt = (
            "You are a helpful assistant trained on Commercial Bank of Ethiopia's FAQs. "
            "Provide clear, concise, and context-aware answers.\n\n"
            f"User: {translated_input.strip()}\nAssistant:"
        )
        english_response = qa_chain.run(prompt)

        # Translate back to original language if needed
        if detected_lang != "en":
            final_response = translator.translate(english_response, dest=detected_lang).text
        else:
            final_response = english_response

        # Display assistant response
        st.chat_message("assistant").markdown(f"💡 **Answer:** {final_response}")

        st.session_state.chat_history.append({
            "user": user_message.strip(),
            "bot": f"💡 **Answer:** {final_response}"
        })

    except Exception as e:
        st.error(f"⚠️ Error generating response: {e}")
