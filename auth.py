import streamlit as st
import hmac
import hashlib
from datetime import datetime
from sheets_db import SheetsDB

def get_password_hash(password):
    """Create a secure hash of the password"""
    salt = "budgetbuddy_salt"  # In production, use a proper salt strategy
    return hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()

def signup_ui():
    """Show signup form and handle signup"""
    st.subheader("Create New Account")
    new_username = st.text_input("Username", key="new_username")
    new_password = st.text_input("Password", type="password", key="new_password")
    new_email = st.text_input("Email", key="new_email")
    
    if st.button("Sign Up"):
        if not new_username or not new_password:
            st.error("Username and password are required")
            return False
        
        db = SheetsDB()
        password_hash = get_password_hash(new_password)
        success, result = db.add_user(new_username, password_hash, new_email)
        
        if success:
            st.success("Account created successfully! Please log in.")
            return True
        else:
            st.error(result)
            return False
    return False

def login_ui():
    """Show login form and handle login"""
    st.subheader("Login to Your Account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        if not username or not password:
            st.error("Username and password are required")
            return False
        
        db = SheetsDB()
        password_hash = get_password_hash(password)
        success, result = db.get_user(username, password_hash)
        
        if success:
            # Store user info in session state
            st.session_state.user_id = result['user_id']
            st.session_state.username = result['username']
            st.session_state.logged_in = True
            return True
        else:
            st.error(result)
            return False
    return False
