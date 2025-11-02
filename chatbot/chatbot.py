# chatbot.py (Optimized for Fast Local Chatbot using Qwen)

import os
import time
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate

# Force CPU-only (prevents GPU memory errors)
os.environ["OLLAMA_NO_GPU"] = "1"

# -------------------------------------------------
# 🧠 LOAD KNOWLEDGE BASE (Cached for Speed)
# -------------------------------------------------
@st.cache_resource
def load_knowledge_base():
    docs = []
    data_folder = os.path.join(os.path.dirname(__file__), "data")

    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)
        if os.path.isfile(path) and filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            for doc in loader.load():
                first_line = doc.page_content.split("\n", 1)[0].strip()
                if first_line.startswith("# USER_TYPE:"):
                    doc.metadata["user_type"] = first_line.replace("# USER_TYPE:", "").strip()
                else:
                    doc.metadata["user_type"] = "General"
                docs.append(doc)

    splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


print("⏳ Loading knowledge base...")
start_time = time.time()
VECTORSTORE = load_knowledge_base()
print(f"✅ Knowledge base loaded in {time.time() - start_time:.2f} seconds")

# -------------------------------------------------
# 💬 CHATBOT RESPONSE FUNCTION
# -------------------------------------------------
def chatbot_response(user_query, city=None, aqi_value=None, category=None, message=None, user_type="General"):
    start = time.time()

    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(user_query)

    filtered_docs = [d for d in docs if d.metadata.get("user_type") in [user_type, "General"]]

    context_text = "\n\n".join([d.page_content for d in filtered_docs])[:2000]

    template = """
You are an AI Air Quality and Health Advisor.
Use the given context to provide helpful, empathetic, and short advice.
Tailor the response for {user_type}.

-------------------
Context:
{context}
-------------------

City: {city}
Current AQI: {aqi_value}
Category: {category}
Summary: {message}

Guidelines:
- Briefly explain the AQI impact.
- Give specific advice for {user_type}.
- Suggest simple daily actions (masks, indoors, hydration).
- Keep it short and clear.
- End with: "This is general informational advice, not medical guidance."

User’s Question:
{query}
"""

    prompt = PromptTemplate(
        input_variables=["context", "city", "aqi_value", "category", "message", "query", "user_type"],
        template=template
    )

    filled_prompt = prompt.format(
        context=context_text,
        city=city or "Unknown",
        aqi_value=aqi_value or "N/A",
        category=category or "N/A",
        message=message or "",
        user_type=user_type,
        query=user_query
    )

    # ✅ Using Qwen locally (CPU / No GPU)
    llm = Ollama(model="qwen2.5:1.5b", temperature=0.3)

    response = llm.invoke(filled_prompt)

    print(f"🕒 Response generated in {time.time() - start:.1f} seconds")
    return response.strip()
