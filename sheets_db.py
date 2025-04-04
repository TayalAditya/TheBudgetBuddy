import gspread
import pandas as pd
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

class SheetsDB:
    def __init__(self):
        self.client = self._get_client()
        self.users_sheet_name = "BudgetBuddy-Users"
        self.transactions_sheet_name = "BudgetBuddy-Transactions"
        
    def _get_client(self):
        # Use Streamlit secrets if available (for deployment)
        if "gcp" in st.secrets:
            creds_dict = st.secrets["gcp"]
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            # Otherwise use local credentials file (for development)
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        
        return gspread.authorize(creds)
    
    def _get_or_create_sheet(self, sheet_name):
        """Get or create a Google Sheet"""
        try:
            return self.client.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            sheet = self.client.create(sheet_name)
            # Make it accessible to anyone with the link
            sheet.share(None, perm_type='anyone', role='reader')
            return sheet
    
    def initialize(self):
        """Set up all necessary sheets and worksheets"""
        # Users sheet
        users_sheet = self._get_or_create_sheet(self.users_sheet_name)
        try:
            users_ws = users_sheet.worksheet("users")
        except gspread.WorksheetNotFound:
            users_ws = users_sheet.add_worksheet(title="users", rows=1000, cols=5)
            users_ws.update('A1:E1', [['user_id', 'username', 'password', 'email', 'created_at']])
        
        # Transactions sheet
        txn_sheet = self._get_or_create_sheet(self.transactions_sheet_name)
        try:
            txn_sheet.worksheet("transactions")
        except gspread.WorksheetNotFound:
            txn_ws = txn_sheet.add_worksheet(title="transactions", rows=1000, cols=7)
            txn_ws.update('A1:G1', [['id', 'user_id', 'date', 'description', 'amount', 'category', 'transaction_type']])
    
    def add_user(self, username, password_hash, email):
        """Add a new user"""
        users_sheet = self.client.open(self.users_sheet_name)
        users_ws = users_sheet.worksheet("users")
        
        # Check if username exists
        usernames = users_ws.col_values(2)[1:]  # Skip header
        if username in usernames:
            return False, "Username already exists"
        
        # Get next user_id
        user_ids = users_ws.col_values(1)[1:]  # Skip header
        next_id = 1 if not user_ids else int(max(user_ids)) + 1
        
        # Add new user
        users_ws.append_row([
            next_id,
            username,
            password_hash,
            email,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
        
        return True, next_id
    
    def get_user(self, username, password_hash):
        """Verify user credentials and return user data"""
        users_sheet = self.client.open(self.users_sheet_name)
        users_ws = users_sheet.worksheet("users")
        
        # Get all users
        try:
            users_data = users_ws.get_all_records()
            for user in users_data:
                if user['username'] == username and user['password'] == password_hash:
                    return True, user
            return False, "Invalid username or password"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def add_transaction(self, user_id, date, description, amount, category, transaction_type):
        """Add a new transaction"""
        txn_sheet = self.client.open(self.transactions_sheet_name)
        txn_ws = txn_sheet.worksheet("transactions")
        
        # Get next transaction ID
        transaction_ids = txn_ws.col_values(1)[1:]  # Skip header
        next_id = 1 if not transaction_ids else int(max(transaction_ids)) + 1
        
        # Add transaction
        txn_ws.append_row([
            next_id,
            user_id,
            date,
            description,
            amount,
            category,
            transaction_type
        ])
        
        return next_id
    
    def get_user_transactions(self, user_id):
        """Get all transactions for a user"""
        txn_sheet = self.client.open(self.transactions_sheet_name)
        txn_ws = txn_sheet.worksheet("transactions")
        
        all_txns = txn_ws.get_all_records()
        user_txns = [txn for txn in all_txns if str(txn['user_id']) == str(user_id)]
        
        return pd.DataFrame(user_txns)
