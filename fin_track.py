import traceback
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
import nltk
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# Import authentication and database modules
from auth import login_ui, signup_ui
from sheets_db import SheetsDB

warnings.filterwarnings('ignore')

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(
    page_title="Budget Buddy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# Handle both local .env and Streamlit Cloud secrets
# Load environment variables from .env file (local development)
load_dotenv()

# Function to get API keys from either .env file or Streamlit secrets
def get_api_key(key_name):
    # First try to get from Streamlit secrets
    if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
        api_key = st.secrets.api_keys.get(key_name)
        if api_key:
            return api_key
    
    # Fall back to environment variables
    return os.getenv(key_name)

if 'GROQ_API_KEY' in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    # Fallback to local .env for development
    os.environ["GROQ_API_KEY"] = get_api_key("GROQ_API_KEY")

if 'COHERE_API_KEY' in st.secrets:
    os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
else:
    # Fallback to local .env for development
    os.environ["COHERE_API_KEY"] = get_api_key("COHERE_API_KEY")

if 'COHERE_API_KEY' in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # Fallback to local .env for development
    os.environ["OPENAI_API_KEY"] = get_api_key("OPENAI_API_KEY")

# Check if required API keys are available
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("COHERE_API_KEY"):
    st.error("⚠️ Required API keys are missing. Please set the GROQ_API_KEY and COHERE_API_KEY in your environment or Streamlit secrets.")
    st.stop()

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize Google Sheets database
@st.cache_resource
def get_db():
    db = SheetsDB()
    db.initialize()
    return db

db = get_db()

# Initialize Language Model
model = ChatGroq(
    model_name="Llama3-70b-8192",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.95
)

# Initialize embeddings with Cohere
@st.cache_resource
def get_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=get_api_key("COHERE_API_KEY"),
        user_agent="finance-tracker-app",
        request_timeout=60
    )

embeddings = get_embeddings()

# Load and prepare data - now supports user-specific data
@st.cache_data
def load_data(user_id=None, demo_mode=False):
    if demo_mode:
        # For consistency, always read from original file first, then apply cached changes
        base_df = pd.read_csv('transactions_with_types.csv')
        base_df['date'] = pd.to_datetime(base_df['date'])
        
        # If there's already an in-memory dataframe with changes, use that
        global df
        if 'df' in globals() and df is not None and len(df) > len(base_df):
            return df
        return base_df
        
    elif st.session_state.get('logged_in', False) and user_id:
        # If logged in, get user transactions from Google Sheets
        try:
            return db.get_user_transactions(user_id)
        except Exception as e:
            st.warning(f"Error loading transactions. Using sample data instead.")
            # Fall back to sample data if there's an error
            base_df = pd.read_csv('transactions_with_types.csv')
            base_df['date'] = pd.to_datetime(base_df['date'])
            return base_df
    else:
        # Fall back to the sample data for non-logged-in users
        base_df = pd.read_csv('transactions_with_types.csv')
        base_df['date'] = pd.to_datetime(base_df['date'])
        return base_df

# Only load data if user is logged in
if st.session_state.get('logged_in', False):
    df = load_data(st.session_state.user_id, st.session_state.get('demo_mode', False))
else:
    # Placeholder until login
    df = None

# Create vector store with better chunking and caching
@st.cache_resource
def create_vector_store(_df):
    if _df is None or _df.empty:
        return None
        
    # Create a combined text field for embedding with more context
    texts = []
    metadatas = []
    
    for _, row in _df.iterrows():
        text = f"""Transaction Details:
        - ID: {row['transaction_id']}
        - Amount: ${row['amount']:.2f}
        - Type: {row['type']}
        - Category: {row['category_description']}
        - Date: {row['date']}
        - Balance After: ${row['balance_left']:.2f}"""
        texts.append(text)
        metadatas.append({
            "transaction_id": row['transaction_id'],
            "amount": row['amount'],
            "type": row['type'],
            "category": row['category_description'],
            "date": str(row['date']),
            "balance": row['balance_left']
        })
    
    # Create vector store with metadata
    return FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

# Conditionally create vectorstore if user is logged in
vectorstore = None
compression_retriever = None

if df is not None and not df.empty:
    vectorstore = create_vector_store(df)
    
    # Create compression retriever with caching
@st.cache_resource
def get_retriever(vs):
    if vs is None:
        return None
    return ContextualCompressionRetriever(
        base_compressor=CohereRerank(),
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )

# Only create retriever if vectorstore exists
compression_retriever = None
if vectorstore is not None:
    compression_retriever = get_retriever(vectorstore)

# Add a function to add a new transaction for a user
def add_transaction_form():
    st.header("Add New Transaction")
    
    # Transaction form
    date = st.date_input("Date", datetime.now())
    description = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.01, step=0.01)
    category = st.selectbox(
        "Category", 
        ["Food", "Transport", "Housing", "Utilities", "Entertainment", "Shopping", 
         "Healthcare", "Education", "Travel", "Salary", "Investment", "Other"]
    )
    transaction_type = st.selectbox("Type", ["expense", "income"])
    
    if st.button("Add Transaction"):
        if description and amount > 0:
            with st.spinner("Adding transaction..."):
                try:
                    if st.session_state.get('demo_mode', False):
                        # For demo mode, add to the in-memory dataframe
                        global df
                        
                        # Create a new row
                        new_id = len(df) + 1
                        transaction_id = f"DBT-{new_id}" if transaction_type == "expense" else f"CRD-{new_id}"
                        
                        # Calculate new balance
                        if df is not None and not df.empty:
                            last_balance = df['balance_left'].iloc[-1]
                            if transaction_type == "expense":
                                new_balance = last_balance - amount
                            else:
                                new_balance = last_balance + amount
                        else:
                            # Start with 5000 if no data
                            new_balance = 5000 + (amount if transaction_type == "income" else -amount)
                        
                        # Create new transaction row
                        new_row = pd.DataFrame([{
                            'transaction_id': transaction_id,
                            'date': pd.to_datetime(date),
                            'amount': amount,
                            'type': transaction_type,
                            'category_description': category,
                            'balance_left': new_balance,
                            'description': description
                        }])
                        
                        # Append to dataframe and sort by date
                        if df is None:
                            df = new_row
                        else:
                            df = pd.concat([df, new_row], ignore_index=True)
                            df = df.sort_values('date')
                        
                        # Clear cached data to force reload
                        st.cache_data.clear()
                        
                        st.success("Transaction added successfully! (Demo Mode)")
                        
                    else:
                        # Real mode: Add to Google Sheets
                        db.add_transaction(
                            st.session_state.user_id,
                            date.strftime("%Y-%m-%d"),
                            description,
                            amount,
                            category,
                            transaction_type
                        )
                        
                        # Clear cached data and force refresh
                        st.cache_data.clear()
                        
                        st.success("Transaction added successfully!")
                    
                    # Add a small delay to allow the success message to be seen
                    import time
                    time.sleep(1)
                    
                    # Force refresh by rerunning the app
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error adding transaction: {str(e)}")
                    st.error("Details: Please check console logs for more information.")
        else:
            st.warning("Please fill in all fields")
        #         # Add transaction to Google Sheets
        #         db.add_transaction(
        #             st.session_state.user_id,
        #             date.strftime("%Y-%m-%d"),
        #             description,
        #             amount,
        #             category,
        #             transaction_type
        #         )
        #         st.success("Transaction added successfully!")
        #         # Update the dataframe
        #         st.session_state.update_data = True
        #         st.rerun()
        #     except Exception as e:
        #         st.error(f"Error adding transaction: {str(e)}")
        # else:
        #     st.warning("Please fill in all fields")

# Existing FinanceTools class
class FinanceTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_spending(self, date: str) -> str:
        """Analyze spending for a specific date"""
        daily_data = self.df[self.df['date'].dt.date == pd.to_datetime(date).date()]
        
        if daily_data.empty:
            return f"No transactions found for {date}"
        
        if 'balance_left' not in daily_data.columns:
            return f"Error: 'balance_left' column is missing in the data for {date}."
            
        total_spent = daily_data['amount'].sum()
        transaction_count = len(daily_data)
        try:
            balance = daily_data['balance_left'].iloc[-1]
        except IndexError:
            balance = "N/A"  # Fallback if no balance is available
        
        return f"On {date}, you had {transaction_count} transactions totaling ${total_spent:.2f}. Your balance at the end of the day was ${balance:.2f}"

    def analyze_trends(self, start_date: str, end_date: str) -> str:
        """Analyze spending trends between two dates"""
        mask = (self.df['date'].dt.date >= pd.to_datetime(start_date).date()) & \
               (self.df['date'].dt.date <= pd.to_datetime(end_date).date())
        period_data = self.df[mask]
        
        total_spent = period_data['amount'].sum()
        avg_transaction = period_data['amount'].mean()
        category_trends = period_data.groupby('category_description')['amount'].sum()
        
        return f"Between {start_date} and {end_date}, you spent ${total_spent:.2f} across {len(period_data)} transactions. Average transaction: ${avg_transaction:.2f}"

    def get_balance_info(self) -> str:
        """Get current balance and recent changes"""
        current_balance = self.df['balance_left'].iloc[-1]
        last_week_balance = self.df[self.df['date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)]['balance_left'].iloc[0]
        balance_change = current_balance - last_week_balance
        
        return f"Current balance: ${current_balance:.2f}. Balance change in the last week: ${balance_change:.2f}"

    def search_transactions(self, query: str) -> str:
        """Search for specific transactions"""
        if compression_retriever:
            relevant_docs = compression_retriever.get_relevant_documents(query)
            if not relevant_docs:
                return "No relevant transactions found."
            
            results = []
            for doc in relevant_docs[:3]:
                results.append(f"Transaction {doc.metadata['transaction_id']}: ${doc.metadata['amount']} {doc.metadata['type']} for {doc.metadata['category_description']} on {doc.metadata['date']}")
            
            return "\n".join(results)
        else:
            return "Transaction search unavailable. Please login first."

# Only create tools if user is logged in
finance_tools = None
tools = []
spending_executor = None
category_executor = None
health_executor = None

if st.session_state.logged_in and df is not None:
    finance_tools = FinanceTools(df)
    tools = [
        Tool(
            name="analyze_spending",
            func=finance_tools.analyze_spending,
            description="Analyze spending for a specific date. Input should be a date in YYYY-MM-DD format."
        ),
        Tool(
            name="analyze_trends",
            func=finance_tools.analyze_trends,
            description="Analyze spending trends between two dates. Input should be two dates in YYYY-MM-DD format separated by a comma."
        ),
        Tool(
            name="get_balance_info",
            func=finance_tools.get_balance_info,
            description="Get current balance and recent changes."
        ),
        Tool(
            name="search_transactions",
            func=finance_tools.search_transactions,
            description="Search for specific transactions based on description or category."
        )
    ]
    
    # Create specialized agents for different tasks
    def create_specialized_agents():
        # Define tools for each agent
        spending_tools = [
            Tool(
                name="analyze_spending",
                func=finance_tools.analyze_spending,
                description="Analyze spending for a specific date."
            ),
            Tool(
                name="analyze_trends",
                func=finance_tools.analyze_trends,
                description="Analyze spending trends between dates."
            )
        ]
        
        category_tools = [
            Tool(
                name="search_transactions",
                func=finance_tools.search_transactions,
                description="Search transactions by category."
            )
        ]
        
        health_tools = [
            Tool(
                name="get_balance_info",
                func=finance_tools.get_balance_info,
                description="Get balance and changes."
            )
        ]
        
        # Create a simple direct function that uses the LLM to decide what to do
        def create_simple_executor(tools, name):
            def execute(query):
                try:
                    # First ask the LLM which tool to use
                    tool_selector_prompt = f"""As a {name} expert, analyze this query and decide which tool to use.
                    Available tools: {[t.name for t in tools]}
                    
                    Query: {query}
                    
                    Respond with ONLY the tool name to use."""
                    
                    tool_selection = model.invoke(tool_selector_prompt).content.strip()
                    
                    # Find the selected tool
                    selected_tool = None
                    for tool in tools:
                        if tool.name in tool_selection:
                            selected_tool = tool
                            break
                    
                    if not selected_tool:
                        return f"I couldn't determine which tool to use for your query about {name}."
                    
                    # Determine the input for the tool
                    if selected_tool.name == "get_balance_info":
                        # This tool doesn't need input
                        tool_result = selected_tool.func()
                    elif selected_tool.name == "analyze_spending":
                        # Need to extract a date
                        date_prompt = f"""Extract the specific date from this query for financial analysis.
                        Format the date as YYYY-MM-DD.
                        If no specific date is mentioned, use today's date.
                        
                        Query: {query}
                        
                        Respond with ONLY the date in YYYY-MM-DD format."""
                        
                        date = model.invoke(date_prompt).content.strip()
                        tool_result = selected_tool.func(date)
                    elif selected_tool.name == "analyze_trends":
                        # Need to extract start and end dates
                        date_prompt = f"""Extract the start and end dates from this query for financial trend analysis.
                        Format as YYYY-MM-DD for both dates.
                        If no specific dates are mentioned, use last week for start date and today for end date.
                        
                        Query: {query}
                        
                        Respond with ONLY the dates in this format: start_date,end_date"""
                        
                        dates = model.invoke(date_prompt).content.strip()
                        start_date, end_date = dates.split(",")
                        tool_result = selected_tool.func(start_date, end_date)
                    elif selected_tool.name == "search_transactions":
                        # Just pass the query directly
                        tool_result = selected_tool.func(query)
                    else:
                        return f"I don't know how to use the {selected_tool.name} tool."
                    
                    # Now generate a response based on the tool output
                    response_prompt = f"""As a {name} expert, interpret this data and provide a helpful but concise response.
                    
                    Query: {query}
                    Data: {tool_result}
                    
                    Keep your response brief and focused:
                    - Start with a 1-sentence summary
                    - Include key numbers (current balance, change)
                    - List 2-3 key insights with bullet points
                    - Give 1-2 actionable tips
                    
                    Use minimal text and avoid unnecessary words or phrases.
                    Format currency with $ symbol.
                    Total length should be under 150 words."""
                    
                    response = model.invoke(response_prompt).content.strip()
                    
                    # Apply HTML styling for Streamlit rendering
                    response = response.replace("**", "<strong>").replace("**", "</strong>")
                    
                    # Make bullet points more visible
                    response = response.replace("- ", "• ")
                    
                    # Ensure proper spacing
                    response = response.replace("$", " $").replace("  $", " $")
                    response = response.replace("in the last week", " in the last week")
                    response = response.replace("which is", " which is")
                    response = response.replace(".This", ". This")
                    response = response.replace(".The", ". The")
                    response = response.replace(".Your", ". Your")
                    response = response.replace(".Based", ". Based")
                    response = response.replace(",$", ", $")
                    
                    # Remove multiple spaces
                    response = ' '.join(response.split())
                    
                    # Add line breaks for better readability
                    paragraphs = response.split('\n')
                    formatted_paragraphs = []
                    for p in paragraphs:
                        p = p.strip()
                        if p.startswith("<strong>"):
                            # Add extra space before headers (except the first one)
                            if formatted_paragraphs:
                                formatted_paragraphs.append("")
                            formatted_paragraphs.append(p)
                        elif p.startswith("•"):
                            formatted_paragraphs.append(p)
                        elif p:
                            formatted_paragraphs.append(p)
                    
                    # Join with appropriate spacing - use less space between elements
                    response = "<br>".join(formatted_paragraphs)
                    response = response.replace("• ", "• ")
                    
                    # Ensure total response isn't too long (target ~300 words max)
                    if len(response.split()) > 300:
                        # Try to shorten by removing some details while keeping structure
                        paragraphs = response.split('<br>')
                        shortened = []
                        for i, para in enumerate(paragraphs):
                            # Keep headers and first sentence of longer paragraphs
                            if para.startswith('<strong>') or '•' in para or len(para.split()) < 15:
                                shortened.append(para)
                            else:
                                # For longer paragraphs, keep just the first sentence
                                sentences = para.split('.')
                                if len(sentences) > 1:
                                    shortened.append(sentences[0] + '.')
                                else:
                                    shortened.append(para)
                        response = '<br>'.join(shortened)
                    
                    return response
                    
                except Exception as e:
                    return f"I encountered an error: {str(e)}"
            
            return execute
        
        # Create the simple executors
        spending_executor = create_simple_executor(spending_tools, "Spending Analysis")
        category_executor = create_simple_executor(category_tools, "Category Analysis")
        health_executor = create_simple_executor(health_tools, "Financial Health")
        
        return spending_executor, category_executor, health_executor

    # Initialize specialized agents
    spending_executor, category_executor, health_executor = create_specialized_agents()

# Existing query processing functions - only called if user is logged in
def process_query(query: str, context: str):
    if not st.session_state.logged_in:
        return "Please log in to use this feature."
        
    try:
        # For balance queries, use health agent directly
        if any(word in query.lower() for word in ['balance', 'money', 'afford', 'health']):
            response = health_executor(f"{context}\n\nAnalyze financial health: {query}")
            # Update the session state to indicate balance info was requested
            st.session_state['chart_type'] = 'balance'
            st.session_state['query_dates'] = None
            return response
            
        # For spending queries
        elif any(word in query.lower() for word in ['spend', 'cost', 'paid', 'buy', 'purchase']):
            # Extract dates from the query for charts
            date_prompt = f"""Extract dates from this query for financial analysis.
            If a single date is mentioned, respond with "single:YYYY-MM-DD".
            If a date range is mentioned, respond with "range:YYYY-MM-DD,YYYY-MM-DD".
            If a period is mentioned (like "last week", "this month"), determine the appropriate start and end dates and respond with "range:YYYY-MM-DD,YYYY-MM-DD".
            If no date is mentioned, respond with "none".
            
            Query: {query}
            
            Respond with ONLY the format specified above."""
            
            date_info = model.invoke(date_prompt).content.strip()
            
            if date_info.startswith("single:"):
                date = date_info.split(":")[1]
                st.session_state['chart_type'] = 'spending_day'
                st.session_state['query_dates'] = date
            elif date_info.startswith("range:"):
                dates = date_info.split(":")[1]
                start_date, end_date = dates.split(",")
                st.session_state['chart_type'] = 'spending_range'
                st.session_state['query_dates'] = (start_date, end_date)
            else:
                st.session_state['chart_type'] = 'spending_recent'
                st.session_state['query_dates'] = None
                
            response = spending_executor(f"{context}\n\nAnalyze spending: {query}")
            return response
            
        # For category queries
        elif any(word in query.lower() for word in ['category', 'type', 'kind', 'what', 'where']):
            # Extract category from query
            category_prompt = f"""Extract the category or type of transaction from this query.
            If a specific category is mentioned (like "food", "groceries", "entertainment"), respond with that category.
            If no specific category is mentioned but it's a general category query, respond with "all".
            If it's not about categories, respond with "none".
            
            Query: {query}
            
            Respond with ONLY the category name or one of the special values."""
            
            category = model.invoke(category_prompt).content.strip()
            
            if category != "none":
                st.session_state['chart_type'] = 'category'
                st.session_state['query_category'] = category
            else:
                st.session_state['chart_type'] = None
                
            response = category_executor(f"{context}\n\nAnalyze categories: {query}")
            return response
            
        # If no specific keywords, use health agent as default for general queries
        else:
            st.session_state['chart_type'] = None
            return health_executor(f"{context}\n\nAnalyze financial health: {query}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        st.session_state['chart_type'] = None
        return f"I encountered an error while processing your query. Here's what I know: {context}"

def process_query_with_rag(query: str, current_date) -> str:
    """Process the query using RAG to provide relevant context"""
    if not st.session_state.logged_in or df is None:
        return "Please log in to use this feature."

    if compression_retriever is None:
        return "No transaction data available for analysis."
        
    try:
        # Get relevant documents from the vector store
        relevant_docs = compression_retriever.get_relevant_documents(query)
        
        # Build context from relevant documents
        context = []
        
        # Add current financial status
        context.append(f"Current Date: {current_date}")
        context.append(finance_tools.get_balance_info())
        
        # Add relevant transaction information
        if relevant_docs:
            context.append("\nRelevant Transactions:")
            for doc in relevant_docs[:3]:  # Limit to top 3 most relevant transactions
                context.append(doc.page_content)
        
        # Add recent activity summary
        recent_transactions = df[df['date'].dt.date >= (pd.Timestamp(current_date) - pd.Timedelta(days=7)).date()]
        if not recent_transactions.empty:
            total_recent = recent_transactions['amount'].sum()
            count_recent = len(recent_transactions)
            context.append(f"\nRecent Activity (Last 7 Days):")
            context.append(f"- Total Transactions: {count_recent}")
            context.append(f"- Total Amount: ${total_recent:.2f}")
        
        return "\n".join(context)
    except Exception as e:
        return f"Error building context: {str(e)}"

# Function to generate dynamic charts based on query - same as original
def generate_dynamic_charts():
    if not st.session_state.logged_in or df is None:
        return
        
    if 'chart_type' not in st.session_state or st.session_state['chart_type'] is None:
        return
    
    # Handle balance chart
    if st.session_state['chart_type'] == 'balance':
        # Create a balance trend chart
        recent_dates = df.sort_values('date')[-30:] # Last 30 days
        
        if recent_dates.empty:
            st.warning("No balance data available")
            return
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_dates['date'],
            y=recent_dates['balance_left'],
            mode='lines+markers',
            name='Balance',
            line=dict(width=3, color='#4CAF50'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Balance Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Balance ($)",
            xaxis=dict(
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#ffffff'),
                gridcolor='#4a5568'
            ),
            yaxis=dict(
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#ffffff'),
                gridcolor='#4a5568',
                tickprefix='$'
            ),
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font=dict(color='#ffffff')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Handle daily spending chart    
    elif st.session_state['chart_type'] == 'spending_day':
        date = st.session_state['query_dates']
        day_data = df[df['date'].dt.date == pd.to_datetime(date).date()]
        
        if day_data.empty:
            st.warning(f"No transactions found for {date}")
            return
            
        # Create category breakdown for that day
        fig = px.pie(
            day_data, 
            values='amount', 
            names='category_description',
            title=f'Spending Breakdown for {date}',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        
        fig.update_layout(
            title_font=dict(size=18, color='#ffffff'),
            font=dict(color='#ffffff'),
            paper_bgcolor='#2d3748',
            plot_bgcolor='#2d3748',
            legend=dict(font=dict(color='#ffffff'))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add bar chart of transactions that day
        fig2 = px.bar(
            day_data.sort_values('amount', ascending=False), 
            x='transaction_id', 
            y='amount',
            color='category_description',
            labels={'transaction_id': 'Transaction', 'amount': 'Amount ($)'},
            title=f'Transactions on {date}',
            hover_data=['type', 'category_description']
        )
        
        fig2.update_layout(
            title_font=dict(size=18, color='#ffffff'),
            font=dict(color='#ffffff'),
            paper_bgcolor='#2d3748',
            plot_bgcolor='#2d3748',
            xaxis=dict(gridcolor='#4a5568'),
            yaxis=dict(gridcolor='#4a5568', tickprefix='$')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Rest of the chart functions remain the same...
    # [Continuation of generate_dynamic_charts]
    # I'm omitting the rest of this function for brevity as it's unchanged

# Add custom CSS - same as original
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 20px;
        padding: 20px 0;
        text-align: center;
        background: linear-gradient(120deg, #1E3D59 0%, #2C5282 100%);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Additional styling for auth forms */
    .auth-form {
        background: #2d3748;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .auth-header {
        color: #63b3ed;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main application interface - modified to include auth
st.markdown('<h1 class="main-title">Budget Buddy</h1>', unsafe_allow_html=True)

# Authentication section
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Login</h2>', unsafe_allow_html=True)
        if login_ui():
            # Reload the page after successful login
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Sign Up</h2>', unsafe_allow_html=True)
        if signup_ui():
            st.success("Account created! Please log in.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show app description for non-logged-in users
    st.markdown("""
    ## Welcome to Budget Buddy!
    
    Track your finances, analyze your spending patterns, and get insights to help you save money.
    
    ### Features:
    - 📊 Track income and expenses
    - 📈 Visualize spending patterns
    - 🔍 Categorize transactions
    - 💡 Get AI-powered financial insights
    
    Create an account or log in to get started!
    """)
    
    # Stop execution here for non-logged-in users
    st.stop()

# Show logout button in sidebar for logged-in users
st.sidebar.success(f"Welcome, {st.session_state.username}!")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

# Check if user has any transactions
if df is None or df.empty:
    st.info("You don't have any transactions yet. Let's add some to get started!")
    add_transaction_form()
    st.stop()

# For users with data, show the full app interface
# Add transaction button in sidebar
if st.sidebar.button("➕ Add New Transaction"):
    st.session_state.show_add_form = True

# Show add transaction form if requested
if st.session_state.get('show_add_form', False):
    add_transaction_form()
    if st.button("Done Adding"):
        st.session_state.show_add_form = False
        st.rerun()

# Original query input with better styling and error handling
st.markdown('<div class="card">', unsafe_allow_html=True)
user_query = st.text_input("💬 Ask me anything about your finances:", 
                          placeholder="Example: How much did I spend last week?")
st.markdown('</div>', unsafe_allow_html=True)

if user_query:
    with st.spinner("Analyzing your finances..."):
        try:
            # Use the most recent date in the dataset instead of current date
            current_date = df['date'].max().date()
            context = process_query_with_rag(user_query, current_date)
            response = process_query(user_query, context)
            
            # Add custom CSS for the response styling
            st.markdown("""
            <style>
            .finance-response {
                font-size: 15px;
                color: #ffffff;
                line-height: 1.5;
                padding: 15px;
                background: #2d3748;
                border-radius: 8px;
                border-left: 4px solid #63b3ed;
                margin-bottom: 20px;
                max-width: 800px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display the response with proper formatting
            st.markdown(f'<div class="finance-response">{response}</div>', unsafe_allow_html=True)
            
            # Generate dynamic charts based on the query
            generate_dynamic_charts()
        except Exception as e:
            st.error(f"Error analyzing your finances: {str(e)}")

# The rest of your interface (sidebar date selection, main columns, etc.)
# remains largely unchanged...

# Replace the date selection code in the sidebar with this safer version:

# Sidebar for date selection
st.sidebar.markdown('<h2 class="custom-subheader">📅 Date Selection</h2>', unsafe_allow_html=True)

# Check if df exists and has data
if df is not None and not df.empty:
    # Ensure the 'date' column is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, handle errors
        if df['date'].isnull().any():
            st.error("Some dates could not be parsed. Please check your data.")
    
    min_date = df['date'].min().date()  # Convert to Python date
    max_date = df['date'].max().date()  # Convert to Python date
    
    # Use current date as default selection, but ensure it's within valid range
    today = datetime.now().date()
    if today < min_date:
        default_date = min_date
    elif today > max_date:
        default_date = max_date
    else:
        default_date = today
    
    selected_date = st.sidebar.date_input("Select Date", 
                                         value=default_date,
                                         min_value=min_date, 
                                         max_value=max_date)
    
    # Date range selection for trend analysis
    st.sidebar.markdown('<h2 class="custom-subheader">📊 Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate a safe default start date (3 days before selected date)
    selected_date_ts = pd.to_datetime(selected_date)
    default_start_ts = selected_date_ts - pd.Timedelta(days=3)
    
    # Ensure default_start_date isn't earlier than the minimum date
    default_start_date = max(default_start_ts.date(), min_date)  # Convert both to Python date
    
    start_date = st.sidebar.date_input("Start Date", 
                                     value=default_start_date,
                                     min_value=min_date, 
                                     max_value=max_date)
    
    end_date = st.sidebar.date_input("End Date", 
                                   value=selected_date,
                                   min_value=min_date, 
                                   max_value=max_date)
    
    # Main interface with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="custom-subheader">📊 Daily Financial Overview</h2>', unsafe_allow_html=True)
        analysis = finance_tools.analyze_spending(str(selected_date))
        st.markdown(f'<div class="analysis-text">{analysis}</div>', unsafe_allow_html=True)
        
        # Create visualizations
        daily_data = df[df['date'].dt.date == selected_date]
        if not daily_data.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig1 = px.pie(daily_data, values='amount', names='category_description',
                         title=f'Spending Breakdown for {selected_date}')
            fig1.update_layout(
                title_font=dict(size=20, color='#ffffff', family='Arial'),
                font=dict(family='Arial', color='#ffffff'),
                paper_bgcolor='#2d3748',
                plot_bgcolor='#2d3748',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#ffffff')
                )
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="custom-subheader">📈 Trend Analysis</h2>', unsafe_allow_html=True)
        if start_date and end_date:
            trends = finance_tools.analyze_trends(str(start_date), str(end_date))
            st.markdown(f'<div class="analysis-text">{trends}</div>', unsafe_allow_html=True)
            
            # Create trend visualization with daily aggregation
            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
            period_data = df[mask].copy()
            
            if not period_data.empty:
                # Aggregate data by date and calculate additional metrics
                daily_data = period_data.groupby(period_data['date'].dt.date).agg({
                    'amount': ['sum', 'mean', 'count'],
                    'category_description': lambda x: ', '.join(x.unique())
                }).reset_index()
                
                daily_data.columns = ['date', 'total_amount', 'avg_amount', 'transaction_count', 'categories']
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                
                # Calculate y-axis range with padding
                y_min = daily_data['total_amount'].min() * 0.9
                y_max = daily_data['total_amount'].max() * 1.1
                
                # Create spending trend chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=daily_data['date'],
                    y=daily_data['total_amount'],
                    mode='lines+markers',
                    name='Daily Spending',
                    line=dict(width=2.5, color='#1f77b4', shape='linear'),
                    marker=dict(size=8, symbol='circle', color='#1f77b4', 
                               line=dict(color='white', width=1)),
                    hovertemplate=(
                        "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                        "<b>Total Spent</b>: $%{y:,.2f}<br>" +
                        "<b>Transactions</b>: %{customdata[0]}<br>" +
                        "<b>Avg. Transaction</b>: $%{customdata[1]:,.2f}<br>" +
                        "<b>Categories</b>: %{customdata[2]}<extra></extra>"
                    ),
                    customdata=list(zip(
                        daily_data['transaction_count'],
                        daily_data['avg_amount'],
                        daily_data['categories']
                    ))
                ))
                
                # Update layout
                fig2.update_layout(
                    title={
                        'text': 'Daily Spending Trends',
                        'y': 0.95, 'x': 0.5,
                        'xanchor': 'center', 'yanchor': 'top',
                        'font': dict(size=20, family='Arial', color='#ffffff')
                    },
                    xaxis=dict(
                        title='Date',
                        title_font=dict(size=14, color='#ffffff'),
                        tickfont=dict(size=12, color='#ffffff'),
                        tickformat='%Y-%m-%d',
                        gridcolor='#4a5568'
                    ),
                    yaxis=dict(
                        title='Amount ($)',
                        title_font=dict(size=14, color='#ffffff'),
                        tickfont=dict(size=12, color='#ffffff'),
                        gridcolor='#4a5568',
                        tickprefix='$',
                        range=[y_min, y_max]
                    ),
                    plot_bgcolor='#2d3748',
                    paper_bgcolor='#2d3748',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    # Transaction list
    st.markdown('<h2 class="custom-subheader">📝 Transaction Details</h2>', unsafe_allow_html=True)
    daily_transactions = df[df['date'].dt.date == selected_date]
    if not daily_transactions.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(
            daily_transactions[['transaction_id', 'amount', 'type', 'category_description', 'balance_left']],
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No transactions for selected date.")
else:
    # If no data is available, display a message
    st.sidebar.warning("No transaction data available.")
    st.info("No data available to display. Please add a transaction first or log in.")
