import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DataCompass",
    page_icon="📊",
    layout="wide"
)

# LLM HANDLING
load_dotenv()
api_key = os.getenv("GROQ_KEY")
if not api_key:
    st.error("GROQ_KEY environment variable not found.")
    st.stop()

client = Groq(api_key=api_key)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
df = None

#Analysis method:
def enhanced_analyze_csv(content, focus_areas=None):
    """Perform advanced statistical analysis on CSV data with optional focus areas."""
    base_prompt = (
        "You are an expert data scientist with deep knowledge in statistical analysis and machine learning. "
        "Analyze the following CSV data and provide: \n"
        "1. Comprehensive statistical summary including measures of central tendency and dispersion\n"
        "2. In-depth analysis of key trends and patterns, including any cyclical or seasonal trends\n"
        "3. Detailed examination of outliers or anomalies, including potential causes and impacts\n"
        "4. Advanced insights for business decisions, including predictive analysis where applicable\n"
        "5. Suggestions for further data collection or analysis that could enhance insights\n"
    )

    if focus_areas:
        base_prompt += f"6. Specific analysis on the following areas of interest: {', '.join(focus_areas)}\n"

    base_prompt += f"\nCSV Data:\n{content}\n"

    prompt = base_prompt + "\nProvide your analysis in a structured, easy-to-read format with clear headings for each section."

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0
    )
    return chat_completion.choices[0].message.content

#Chat method
def enhanced_chat_with_csv(query, context, previous_conversation=None):
    """Enhanced chat with CSV data using natural language and conversation history."""
    system_message = (
        "You are a highly precise data analyst assistant with expertise in statistics and business intelligence. "
        "Your role is to provide accurate, insightful answers based solely on the provided data context. "
        "Always cite specific data points in your answers and express any uncertainties clearly. "
        "If the provided context is insufficient to answer a question accurately, state this explicitly "
        "and suggest what additional information would be needed to provide a complete answer."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context:\n{context}"}
    ]

    if previous_conversation:
        messages.extend(previous_conversation)

    messages.append({"role": "user", "content": query})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
        temperature=0
    )
    return chat_completion.choices[0].message.content

#Vector Database
def create_faiss_index(dataframe):
    """Create and store embeddings in a FAISS index."""
    global index
    with st.spinner('Creating embeddings for vector database...'):
        # Initialize progress bar
        progress_bar = st.progress(0)
        embeddings = []
        total_rows = len(dataframe)
        
        # Create embeddings with progress tracking
        for i, row in enumerate(dataframe.values):
            embedding = embedding_model.encode(str(row))
            embeddings.append(embedding)
            
            # Update progress bar
            progress = (i + 1) / total_rows
            progress_bar.progress(progress)
            
        # Convert to numpy array
        embeddings = np.array(embeddings)

#Relevant Context
def get_relevant_context(query, k=5):
    """Retrieve relevant rows based on query similarity."""
    try:
        with st.spinner('Processing query and searching vector database...'):
            # Create query embedding
            query_embedding = embedding_model.encode(query).reshape(1, -1)
            
            # Search for similar vectors
            distances, indices = index.search(query_embedding, k)
            
            # Get matching rows and add similarity scores
            results = df.iloc[indices[0]].copy()
            results['similarity_score'] = 1 / (1 + distances[0])  # Convert distance to similarity score
            
            # Sort by similarity score
            results = results.sort_values('similarity_score', ascending=False)
        
        return results
    
    except Exception as e:
        st.error(f"Error searching vector database: {str(e)}")
        return pd.DataFrame()


#Visualization method
def create_advanced_visualization(data, viz_type, x_col, y_col, color_col=None, 
                                customization_options=None):
    """Create advanced visualizations with detailed customization options."""
    
    # Set default customization options if none provided
    if customization_options is None:
        customization_options = {
            'title': f"{viz_type} of {y_col if y_col else x_col} by {x_col}",
            'template': "plotly_white",
            'height': 1000,
            'width': 1200,  # Auto-width
            'opacity': 0.7,
            'trendline': False,
            'marginal': None,
            'animation_frame': None,
            'log_scale': False
        }

    fig = None
    
    if viz_type == "Scatter Plot":
        fig = px.scatter(
            data, x=x_col, y=y_col, color=color_col,
            opacity=customization_options['opacity'],
            trendline='ols' if customization_options['trendline'] else None,
            marginal_x='histogram' if customization_options['marginal'] else None,
            marginal_y='histogram' if customization_options['marginal'] else None,
            animation_frame=customization_options['animation_frame']
        )
        
    elif viz_type == "Line Plot":
        fig = px.line(
            data, x=x_col, y=y_col, color=color_col,
            line_shape='linear',
            render_mode='svg'
        )
        if customization_options['log_scale']:
            fig.update_yaxes(type='log')
            
    elif viz_type == "Bar Chart":
        fig = px.bar(
            data, x=x_col, y=y_col, color=color_col,
            barmode='group',
            opacity=customization_options['opacity']
        )
        
    elif viz_type == "Box Plot":
        fig = px.box(
            data, x=x_col, y=y_col, color=color_col,
            points='outliers',  # Show outlier points
            notched=True  # Add confidence intervals
        )
        
    elif viz_type == "Violin Plot":
        fig = px.violin(
            data, x=x_col, y=y_col, color=color_col,
            box=True,  # Add box plot inside violin
            points='outliers'  # Show outlier points
        )
        
    elif viz_type == "Histogram":
        fig = px.histogram(
            data, x=x_col, color=color_col,
            marginal='box',  # Add box plot on marginal
            opacity=customization_options['opacity']
        )
        
    elif viz_type == "Correlation Heatmap":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu",
            aspect='auto'
        )
        # Add correlation values as text
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    x=i, y=j,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
                )
                
    elif viz_type == "Density Contour":
        fig = px.density_contour(
            data, x=x_col, y=y_col,
            marginal_x="histogram",
            marginal_y="histogram"
        )
        
    elif viz_type == "3D Scatter":
        z_col = customization_options.get('z_column')
        if z_col:
            fig = px.scatter_3d(
                data, x=x_col, y=y_col, z=z_col,
                color=color_col,
                opacity=customization_options['opacity']
            )

    # Apply common layout updates
    if fig:
        fig.update_layout(
            title=customization_options['title'],
            template=customization_options['template'],
            height=customization_options['height'],
            width=customization_options['width'],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add hover templates
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "%{y}<br>" +
                         "<extra></extra>"
        )

    return fig

#Advanced summary of visualization
def get_advanced_summary_stats(data, column):
    """Get detailed summary statistics for a numeric column."""
    stats = data[column].describe()
    additional_stats = pd.Series({
        'kurtosis': data[column].kurtosis(),
        'skewness': data[column].skew(),
        'variance': data[column].var(),
        'distinct_values': data[column].nunique(),
        'missing_values': data[column].isnull().sum(),
        'missing_percentage': (data[column].isnull().sum() / len(data)) * 100
    })
    return pd.concat([stats, additional_stats])

# Streamlit UI
st.title("DataCompass: \nAnalyse 🔎 Chat 💬 and Visualize 📊 your Data")

# File handling
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Loading and processing the CSV file...'):
        df = pd.read_csv(uploaded_file)
        create_faiss_index(df)

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Visualization", "Chat Interface"])

    with tab1:
        st.header("Data Analysis")
        with st.spinner('Loading data preview...'):
            st.dataframe(df.head())
        
        if st.button("Perform Analysis"):
            with st.spinner('Analyzing data...'):
                analysis = enhanced_analyze_csv(df.head(50).to_string())
                st.write(analysis)

    with tab2:
        st.header("Data Visualization")
        
        # Visualization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", 
                 "Histogram", "Correlation Heatmap"]
            )
        
        with col2:
            with st.spinner('Processing column types...'):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                categorical_columns = df.select_dtypes(include=['object']).columns
            
            x_col = st.selectbox("Select X-axis", df.columns)
        
        with col3:
            if viz_type != "Histogram" and viz_type != "Correlation Heatmap":
                y_col = st.selectbox("Select Y-axis", numeric_columns)
            else:
                y_col = None

        # Optional color variable
        color_col = st.selectbox("Select Color Variable (optional)", 
                               ['None'] + list(categorical_columns))
        color_col = None if color_col == 'None' else color_col

        # Create and display visualization
        with st.spinner('Creating visualization...'):
            fig = create_advanced_visualization(df, viz_type, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        if st.checkbox("Show Summary Statistics"):
            if viz_type != "Correlation Heatmap":
                st.write("Summary Statistics:")
                selected_col = y_col if y_col else x_col
                if selected_col in numeric_columns:
                    with st.spinner('Calculating summary statistics...'):
                        st.dataframe(get_advanced_summary_stats(df, selected_col))

    with tab3:
        st.header("Chat with your CSV")
        query = st.text_input("Ask a question about your data:")
        
        if query:
            with st.spinner('Finding relevant context...'):
                relevant_context = get_relevant_context(query)
                st.subheader("Relevant Context:")
                st.dataframe(relevant_context)
            
            if st.button("Get Answer"):
                with st.spinner('Generating response...'):
                    response = enhanced_chat_with_csv(query, relevant_context.to_string())
                    st.write("Answer:")
                    st.write(response)

# sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This application provides three main features:
    1. **Data Analysis**: Automated analysis of your CSV data
    2. **Data Visualization**: Interactive plots and charts
    3. **Chat Interface**: Ask questions about your data in natural language
    """)
    
    st.header("Visualization Tips")
    st.write("""
    - **Scatter Plot**: Good for showing relationships between two variables. Use animation for time series.
    - **Line Plot**: Best for time series or trend data
    - **Bar Chart**: Perfect for comparing categories
    - **Box Plot**: Shows distribution and outliers with confidence intervals
    - **Violin Plot**: Displays full distribution shape with embedded box plot
    - **Histogram**: Shows distribution of a single variable with optional marginal plots
    - **Correlation Heatmap**: Shows relationships between all numeric variables with annotations
    - **Density Contour**: Reveals concentration patterns in 2D relationships
    - **3D Scatter**: Visualize relationships between three variables
    """)
    
    st.header("Usage Tips")
    st.write("""
    - Upload a CSV file to begin
    - For analysis, click 'Perform Analysis' to get insights
    - For visualization, select plot type and variables
    - For chat, type your question and click 'Get Answer'
    """)