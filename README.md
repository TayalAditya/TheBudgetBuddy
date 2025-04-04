# BudgetBuddy - AI-Powered Financial Assistant 💰🤖

**BudgetBuddy** is an intelligent, AI-integrated budgeting application that helps you track, manage, and understand your personal finances using natural language queries. With an interactive UI and seamless backend integration, it turns expense tracking into an intuitive and insightful experience.

## 🚀 Features

- 🔍 **Natural Language Queries** — Ask questions about your finances in plain English
- 📊 **Dynamic Visualizations** — Interactive charts for spending trends and breakdowns
- 📁 **Personalized Budget Tracking** — Categorize transactions, monitor balances
- 🔐 **Secure Login System** — User-based access with authentication
- 📈 **Financial Insights** — Understand patterns and manage funds better

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Flask-style architecture)
- **AI Integration**:
  - Groq LLM (Llama3-70b-8192)
  - Cohere Embeddings + Reranking
  - LangChain for orchestration
- **Data Handling**: Pandas, FAISS for vector search
- **Visualization**: Plotly
- **Auth**: JWT-style login system (via `auth.py`)

## ⚙️ Setup Instructions

### 📋 Prerequisites

- Python 3.8+
- API Keys:
  - `GROQ_API_KEY`
  - `COHERE_API_KEY`
  - `OPENAI_API_KEY`

### 🧪 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TayalAditya/TheBudgetBuddy.git
   cd TheBudgetBuddy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your API keys:
   ```makefile
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
   Replace `your_groq_api_key`, `your_cohere_api_key`, and `your_openai_api_key` with your actual API keys.
4. Run the Streamlit application:
   ```bash
   streamlit run fin_track.py
   ```

## ☁️ Deploy on Streamlit Cloud

To deploy your application on Streamlit Cloud, follow these steps:

1. Push your code to GitHub.
2. Log in to Streamlit Cloud.
3. Create a new app, linking this repository.
4. In the app’s Advanced settings, add the following environment variables:
	* `GROQ_API_KEY`
	* `COHERE_API_KEY`
	* `OPENAI_API_KEY`
5. Click Deploy to go live!

## 📂 File Structure

The project has the following file structure:
```plaintext
.
├── .devcontainer/               # Dev container configuration
├── README.md                    # This file
├── auth.py                      # Authentication logic
├── fin_track.py                 # Main Streamlit application
├── sheets_db.py                 # Sheets / DB interactions
├── transactions_with_types.csv  # Sample transaction data
├── requirements.txt             # Project dependencies
├── setup.py                     # Package setup script
```

## 🎯 Contributions & Feedback

Contributions, feedback, and suggestions are always welcome! If you find this project helpful, consider giving it a ⭐ on GitHub.

## 📜 License

This project is licensed under the MIT License.

## Made with ❤️ by

Made with ❤️ by @TayalAditya
