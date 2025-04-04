
This project was built under Xpecto 2025 Competition FrostHack by team Code Architects with members Aditya Tayal, Arendra Kumar, Siddhanth Vashist, Vinamra Garg

# BudgetBuddy - Real-Time AI-Powered Finance Tracker 💰📊
**BudgetBuddy** is a real-time, AI-powered personal finance assistant that helps you **track spending**, **analyze budgets**, and **gain smart insights** through natural language. Each user gets a secure, personalized dashboard that remembers financial history and dynamically updates based on real-time queries.

## 🚀 Key Features
- 🔐 **Secure Signup & Login** — Personalized access with JWT tokens  
- 🔁 **Real-Time Transaction Syncing** — Ingested via Pathway vector store  
- 🧠 **AI Assistant** — Powered by Fetch AI, LangChain & Groq LLM  
- 💬 **Smart Queries** — Ask questions like "What’s my balance?" or "Can I afford a ₹500 dinner?"  
- 📅 **Date-wise Breakdown** — Visualizes day-wise spending with interactive charts  
- 📊 **Chart Click Filters** — Click any category to view detailed breakdown  
- 🔄 **Google Sheets Integration (Planned)** — Recollect past user data from synced sheets  
- 🌐 **Simple Web Interface** — Built with Gradio for ease of use  

## 🛠️ Tech Stack
- **Frontend:** Gradio  
- **AI/LLM:** Groq LLM (Llama3-70B), Cohere Embeddings + Reranking  
- **Agentic RAG:** Fetch AI AutonomousAgent + LangChain tools  
- **Vector Store:** Pathway’s dynamic vector database  
- **Data Handling:** Pandas, JWT Auth  
- **Visualization:** Plotly  
- **Authentication:** `auth.py` using JWT tokens  

## ⚙️ Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/TayalAditya/TheBudgetBuddy.git
cd TheBudgetBuddy
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add environment variables in a `.env` file
```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 4. Run the application
```bash
streamlit run fin_track.py
```

## ☁️ Deploy on Streamlit Cloud
1. Push your code to GitHub  
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)  
3. Create a new app from this repo  
4. In *Advanced Settings*, add these environment variables:
   - `GROQ_API_KEY`
   - `COHERE_API_KEY`
   - `OPENAI_API_KEY`
5. Hit **Deploy** 🚀

## 📂 File Structure
```plaintext
.
├── .devcontainer/               # Dev container configs
├── README.md                    # You're reading it!
├── auth.py                      # Signup/Login & JWT handling
├── fin_track.py                 # Main Gradio UI & RAG pipeline
├── sheets_db.py                 # Handles Sheets data (if integrated)
├── transactions_with_types.csv  # Sample transaction data
├── requirements.txt             # Project dependencies
├── setup.py                     # Packaging (if needed)
```

## 🧠 How RAG Works Here
BudgetBuddy uses **Retrieval-Augmented Generation (RAG)** to fetch relevant transactions from a real-time vector index (via Pathway), then passes them to an **AI agent (Fetch AutonomousAgent)** that crafts responses or visual explanations. You can query your budget naturally and receive charts or breakdowns — all on the fly.

## ✅ Examples of Supported Queries
- "How much did I spend this week?"  
- "What were my top 3 categories last month?"  
- "Can I afford a ₹10,000 laptop?"  
- "Show me what I spent on groceries last Sunday"  

## 🎯 Contributions & Feedback
Got ideas? Bugs? Drop an issue or a pull request!  
If you like the project, leave a ⭐ to show support.

---
Made with ❤️ by [Team Code Architects](https://github.com/TayalAditya)
