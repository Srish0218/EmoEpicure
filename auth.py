# login_signup.py

import streamlit as st
from login import login
from signup import signup
from main_app import main_app

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon=":snowflake:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False

    if not st.session_state.user_authenticated:
        login_signup_page()
    else:
        main_app()

def login_signup_page():
    st.title("Cafe Owner Authentication")
    f_sign, f_login = st.columns(2)

    with f_sign:
        st.info("Signup Form")
        signup_form = st.form(key="signup_form")
        new_username = signup_form.text_input("Username")
        new_restaurant_name = signup_form.text_input("Restaurant Name")
        new_password = signup_form.text_input("Password", type="password")
        submit_signup = signup_form.form_submit_button("Sign Up")

    with f_login:
        st.info("Login form")
        login_form = st.form(key="login_form")
        username = login_form.text_input("Username")
        restaurant_name = login_form.text_input("Restaurant Name")
        password = login_form.text_input("Password", type="password")
        submit_login = login_form.form_submit_button("Login")

    if submit_signup:
        signup(new_username, new_password, new_restaurant_name)
        st.success("Account created! Please log in.")
    elif submit_login:
        if login(username, password):
            st.session_state.user_authenticated = True
            st.success("Successfully authenticated!")
        else:
            st.error("Authentication failed. Please check your credentials.")

if __name__ == "__main__":
    main()
