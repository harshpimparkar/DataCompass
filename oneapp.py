#Not to be used
#with visualization
import os
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import plotly.express as px

# Load environment variables from .env file if present
load_dotenv()

# Initialize the Groq Cloud client using the API key from environment variables
api_key = os.getenv("GROQ_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable not found.")
    st.stop()

client = Groq(api_key=api_key)

# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = None
df = None

def analyze_with_llm(content):
    """Analyze the CSV data using Groq LLM with a custom prompt."""
    prompt = (
        "You are a data analyst. Analyze the following CSV data and extract meaningful insights. "
        "Summarize trends, important statistics, or any interesting patterns you can identify, "
        "and present your findings as a set of concise and readable paragraphs.\n\n"
        f"CSV Data:\n{content}\n"
    )

    # Send the request to the Groq Cloud LLM
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"  # Specify the model being used
    )

    # Access the generated result (modify according to actual response structure)
    result = chat_completion.choices[0].message.content
    return result

def create_faiss_index(dataframe):
    """Create and store embeddings in a FAISS index from the provided DataFrame."""
    global index
    # Generate embeddings for each row in the DataFrame
    embeddings = np.array([embedding_model.encode(str(row)) for row in dataframe.values])
    # Create a FAISS index using L2 distance
    index = faiss.IndexFlatL2(embeddings.shape[1])  
    index.add(embeddings)  # Add the embeddings to the index

# Title of the Streamlit app
st.title("CSV File Analysis with LLMs (Groq Cloud)")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV file:")
    st.write(df)

    # Create FAISS index for the uploaded DataFrame
    create_faiss_index(df)

    # Input box for analysis queries
    query = st.text_input("Enter your analysis query:")

    # Button to analyze relevant data with the LLM
    if st.button("Analyze Data with LLM"):
        if query:
            with st.spinner('Analyzing your data...'):
                # Convert the entire DataFrame to string format
                csv_chunk = df.to_string()
                analysis_result = analyze_with_llm(csv_chunk)
                st.write("Analysis Result:")
                st.write(analysis_result)
        else:
            st.error("Please enter a query for analysis.")

    # Visualization Section
    st.header("Generate Visualizations")

    if st.button("Generate Visualizations"):
        with st.spinner('Generating visualizations...'):
            # Select numeric columns for visualization
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numeric_cols) > 0:
                # Generate and display a histogram for the first numeric column
                hist_fig = px.histogram(df, x=numeric_cols[0], title=f"Histogram of {numeric_cols[0]}")
                st.plotly_chart(hist_fig)

                # If there are at least two numeric columns, generate a scatter plot
                if len(numeric_cols) > 1:
                    scatter_fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                             title=f"Scatter Plot of {numeric_cols[0]} vs {numeric_cols[1]}")
                    st.plotly_chart(scatter_fig)
            else:
                st.warning("No numeric columns found for visualization.")
