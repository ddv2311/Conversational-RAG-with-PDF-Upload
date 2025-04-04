import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational-RAG with PDF upload and chat history")
st.write("Upload a PDF file and ask questions about its content. The app will remember your previous questions and answers.")

api_key = st.text_input("Enter your Groq API key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
    session_id = st.text_input("Session ID", value="default_session")
    
    # Statefully managing chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
    # Initialize chat history display
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Process uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
    
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and a latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as it is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer prompt
        system_prompt = (
            "You are an assistant that answers questions based on the provided context. "
            "If the question is not answerable based on the context, "
            "say 'I don't know'. "
            "If the question is not related to the context, "
            "say 'This question is not related to the context.' "
            "Answer length should be as asked by the user."
            "\n\n"
            "Context:\n{context}\n\n"
        )        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF content"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get and display AI response
            with st.chat_message("assistant"):
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                st.markdown(response["answer"])
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

else:
    st.warning("Please enter your Groq API key to continue.")