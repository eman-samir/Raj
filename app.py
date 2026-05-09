import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


#  Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

#  UI
st.set_page_config(page_title="Architecture", page_icon="🏛️")

st.image("logo.png", width=200)
st.title("The Intelligent Architecture Assistant")
st.write("Ask anything about architecture!")

#  Chat history memory
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
# Load embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)

# Load vectorstore
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
# User input
query = st.chat_input("Enter your question:")

if query:
      # Save & show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
     #RAG retrieval
    docs = db.similarity_search(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

   
    #  Prompt
    prompt = f"""
    Answer ONLY using this context:

    {context}

    Question: {query}
    """

    #  Gemini response (with fallback)
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = " Gemini quota issue. Try again later."

    #  Save & show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


        with st.expander("Sources"):
            for i, doc in enumerate(docs[:3]):
                meta = doc.metadata
       
                author = meta.get("author", "Unknown Author")
                year = meta.get("year", "n.d.")
                title = meta.get("title", "Untitled")
                source = meta.get("source", "Retrieved from database")

        #  APA style citation
                citation = f"{author} ({year}). {title}. {source}"

                st.write(f"Source {i+1}: {citation}")
        #  Show sources
      #  with st.expander("Sources"):
       #     for i, doc in enumerate(docs[:3]):
       #         st.write(f"Source {i+1}:")
       #         st.write(doc.page_content[:300] + "...")
