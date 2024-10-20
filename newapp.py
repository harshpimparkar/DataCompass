#Not to be used
import streamlit as st
import os
import groq
import pandas as pd
import tempfile
from pathlib import Path
from typing import List, Tuple
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from dotenv import load_dotenv

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Chat with CSV",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

class CSVChatApp:
    def __init__(self):
        self.api_key = os.getenv("GROQ_KEY")
        if not self.api_key:
            st.error("Please set your GROQ API key in the environment variables.")
            st.stop()
        
        # Set API key for GROQ
        os.environ["GROQ_KEY"] = self.api_key
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'chain' not in st.session_state:
            st.session_state.chain = None

    def process_csv(self, file_path) -> List[Document]:
        """
        Process the uploaded CSV file and convert it to LangChain documents.
        
        Args:
            file_path: Streamlit UploadedFile object
        Returns:
            List of LangChain Document objects
        """
        try:
            # Create a temporary file using tempfile module
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                # Write the uploaded file content to the temporary file
                tmp_file.write(file_path.getvalue())
                tmp_file_path = tmp_file.name

            # Display the DataFrame
            df = pd.read_csv(tmp_file_path)
            st.write("Uploaded CSV file:")
            st.write(df)
            
            # Load data into LangChain documents
            loader = CSVLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"Note: Temporary file cleanup failed: {str(e)}")
        
        return chunks

    def setup_chain(self, chunks: List[Document]):
        """
        Set up the conversation chain with GROQ and FAISS.
        
        Args:
            chunks: List of document chunks
        """
        try:
            # Initialize embeddings with explicit model name
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Initialize GROQ model
            llm = ChatGroq(
                temperature=0.5,
                model_name="llama-3.1-70b-versatile",
                groq_api_key=self.api_key
            )
            
            # Create the conversational chain
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
            )
        except Exception as e:
            st.error(f"Error setting up the chain: {str(e)}")
            raise

    def get_response(self, question: str) -> str:
        """
        Get response from the chain for a given question.
        
        Args:
            question: User's question
        Returns:
            Model's response
        """
        try:
            response = st.session_state.chain({
                "question": question,
                "chat_history": st.session_state.chat_history
            })
            return response['answer']
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return "Sorry, I encountered an error processing your question. Please try again."

    def run(self):
        """Run the Streamlit application."""
        st.header("Chat with CSV")
        
        user_csv = st.file_uploader("Upload your CSV file here", type=["csv"])
        
        if user_csv is not None:
            try:
                chunks = self.process_csv(user_csv)
                self.setup_chain(chunks)
                
                user_question = st.text_input("Ask your query:")
                
                if st.button("Get Answer"):
                    if user_question:
                        with st.spinner('Fetching response...'):
                            response = self.get_response(user_question)
                            st.write("Response:")
                            st.write(response)
                            
                            # Update chat history
                            st.session_state.chat_history.append(
                                (user_question, response)
                            )
                    else:
                        st.error("Please enter a question.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = CSVChatApp()
    app.run()