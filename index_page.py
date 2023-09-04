import sqlite3
import os
import streamlit as st
import mysql.connector
from urllib.parse import urlencode
# =========================================   Create a connection to the database  ===============================
# pip install mysql-connector-python
# Establish a connection to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="deeplearning"
)

# Create a cursor object
mycursor = mydb.cursor()


st.sidebar.title("Login & Registeration ")
login_option = st.sidebar.selectbox("Menu" , ("Home" , "Register" , "Login"))

#This is load the markdown page for the entire home page
def get_file_content_as_string(path_1):
    path = os.path.dirname(__file__)
    my_file = path + path_1
    with open(my_file,'r') as f:
        instructions=f.read()
    return instructions

# Main_image = st.image('https://imgur.com/3UtpvAy.png',caption='Source: https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview ')
header_txt = st.markdown(get_file_content_as_string('/instructions2.md'), unsafe_allow_html=True)
Main_image = st.image('The-structures-of-different-deep-learning-models.png', caption='Neural Networks Can Solve Almost Anything')
readme_text = st.markdown(get_file_content_as_string('/Instructions.md'), unsafe_allow_html=True)

def login_function( ):
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()
    st.title("Login To Access Our System")
    # Define the login form fields
    login_email = st.text_input("Enter Email")
    login_password = st.text_input("Enter Password", type="password")
    user_level = st.selectbox("Who Are You?", ("", "Admin", "Radiologist", "Doctor", "Patient"))

    # Define the login form submission action
    if st.button("Login"):
        #erase the main page contents first
        if user_level == "Admin":
            # Check if the email and password match any of the user records in the database
            mycursor.execute("SELECT * FROM admin WHERE email = %s AND password = %s", (login_email, login_password))
            user = mycursor.fetchone()

            if user:
                st.success("Login successful!")
                st.session_state["user_id"] = user[0]  # Assuming the first column is user_id
                st.session_state["user_level"] = user_level
                params = urlencode({"user_id": user[0]})
                url = f"http://localhost:8503/?{params}"
                st.experimental_set_query_params()
                st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
            else:
                st.error("Incorrect email or password.")
        elif user_level == "Radiologist":
            # Check if the email and password match any of the user records in the database
            mycursor.execute(
                "SELECT * FROM radiologist WHERE email = %s AND password = %s", (login_email, login_password)
            )
            user = mycursor.fetchone()

            if user:
                st.success("Login successful!")
                st.session_state["user_id"] = user[0]  # Assuming the first column is user_id
                st.session_state["user_level"] = user_level
                params = urlencode({"user_id": user[0]})
                url = f"http://localhost:8505/?{params}"
                st.experimental_set_query_params()
                st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
            else:
                st.error("Incorrect email or password.")
        elif user_level == "Doctor":
            # Check if the email and password match any of the user records in the database
            mycursor.execute("SELECT * FROM doctor WHERE email = %s AND password = %s", (login_email, login_password))
            user = mycursor.fetchone()
            if user:
                st.success("Login successful!")
                st.session_state["user_id"] = user[0]  # Assuming the first column is user_id
                st.session_state["user_level"] = user_level
                params = urlencode({"user_id": user[0]})
                url = f"http://localhost:8504/?{params}"
                st.experimental_set_query_params()
                st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
            else:
                st.error("Incorrect email or password.")

        elif user_level == "Patient":
            # Check if the email and password match any of the user records in the database
            mycursor.execute("SELECT * FROM patients WHERE email = %s AND password = %s", (login_email, login_password))
            user = mycursor.fetchone()

            if user:
                st.success("Login successful!")
                st.session_state["user_info"] = {"user_id": user[0], "user_level": user_level}
                url = "http://localhost:8502"
                st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
            else:
                st.error("Incorrect email or password.")
        # Redirect to the specified URL





def register_function():
        header_txt.empty()
        Main_image.empty()
        readme_text.empty()
        st.title("Register In To Our System")
        # Define the form fields
        register_name = st.text_input("Name")
        register_email = st.text_input("Email")
        register_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")


        # Define the form submission action
        if st.button("Register"):
            #erase the main page contents first
            if register_password == confirm_password:
                # Save the form data to the database
                mycursor.execute("INSERT INTO patients (name, email, password) VALUES (%s, %s, %s)", (register_name, register_email, register_password))
                mydb.commit()
                st.success("Registration successful!")
                st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
                #url = "http://localhost:8502"
                #st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
                #st.button("Click [here](http://localhost:8502/) to login.")

            else:
                st.error("Passwords do not match.")


if login_option == "Login":
    # calling the login function
    login_function()

if login_option == "Register":
    # calling the register function
    register_function()



#  ====================================== The Second Side Bar =======================================================================
st.sidebar.markdown("---")
st.sidebar.title(" Account Facilities ")
login_option2 = st.sidebar.selectbox("Configure Account" , ("" ,"Update Account" , "Delete Account"))

def update_account():
    st.title("Update Your Account")

    # Define the login form fields
    login_email = st.text_input("Enter Email")
    login_password = st.text_input("Enter Password", type="password")
    user_level = st.selectbox("Who Are You?", ("", "Admin", "Radiologist", "Doctor", "Patients"))

    if user_level:
        # Select the right table according to user_level
        mycursor.execute(f"SELECT * FROM {user_level.lower()} WHERE email = %s AND password = %s", (login_email, login_password))

        user = mycursor.fetchone()

        if user:
            st.title("Update Your Old Information")

            new_name = st.text_input("New Name", value=user[1])
            new_email = st.text_input("New Email", value=user[2])
            new_password = st.text_input("New Password", type="password")

            if st.button("Submit"):
                mycursor.execute(f"UPDATE {user_level.lower()} SET name = %s, email = %s, password = %s WHERE password = %s",
                                (new_name, new_email, new_password, login_password))
                mydb.commit()
                st.success("Your information has been updated.")
        else:
            st.error("Invalid email or password. Please try again.")
    else:
        st.info("Please select your user level first.")


def delete_account():

    st.title("Delete Your Account")

    # Define the login form fields
    login_email = st.text_input("Enter Email")
    login_password = st.text_input("Enter Password", type="password")
    user_level = st.selectbox("Who Are You?", ("", "Admin", "Radiologist", "Doctor", "Patients"))

    if user_level:
        # Select the right table according to user_level
        mycursor.execute(f"SELECT * FROM {user_level.lower()} WHERE email = %s AND password = %s", (login_email, login_password))

        user = mycursor.fetchone()

        if user:
            st.title("Delete Your Information")

            if st.button("Delete Account"):
                mycursor.execute(f"DELETE FROM {user_level.lower()} WHERE password = %s", (login_password,))
                mydb.commit()

                st.success("Your account has been deleted.")
                st.stop()
        else:
            st.error("Invalid email or password. Please try again.")
    else:
        st.info("Please select your user level first.")


if login_option2 == "Update Account":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()
    update_account()

if login_option2 == "Delete Account":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()
    delete_account()
