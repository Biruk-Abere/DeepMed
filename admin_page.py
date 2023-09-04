import streamlit as st
import mysql.connector
from streamlit_option_menu import option_menu
import pandas as pd
import os


# connect to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="deeplearning"
)

# Create a cursor object
mycursor = mydb.cursor()

#This is load the markdown page for the entire home page
def get_file_content_as_string(path_1):
    path = os.path.dirname(__file__)
    my_file = path + path_1
    with open(my_file,'r') as f:
        instructions=f.read()
    return instructions


# Main_image = st.image('https://imgur.com/3UtpvAy.png',caption='Source: https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview ')
header_txt = st.markdown(get_file_content_as_string('/instructions3.md'), unsafe_allow_html=True)
Main_image = st.image('system_admin.jpg', caption='Neural Networks Can Solve Almost Anything')
#readme_text = st.markdown(get_file_content_as_string('/Instructions4.md'), unsafe_allow_html=True)

st.sidebar.title("The Admin Page")
login_option3 = st.sidebar.selectbox("Menu" , ("" , "Manage User" , "Manage Radiologists" , "Manage Doctors" , "Logout"))
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")

def manage_users():
    # create a function to fetch all the users from the database
    def fetch_users():
        mycursor.execute("SELECT * FROM patients")
        users = mycursor.fetchall()
        return users

    # create a function to delete a user from the database
    def delete_user(email):
        mycursor.execute("DELETE FROM patients WHERE email = %s", (email,))
        mydb.commit()

    # fetch the list of users from the database
    users = fetch_users()

    # create a pandas DataFrame to hold the user data
    df = pd.DataFrame(users, columns=["Id" , "Name", "Email", "Password", "Gender", "Age", "Symptoms", "Prediction", "Feedback", "Treatment_Plan", "Image", "Category", "Confidence", "Prescribed Medication", "Dosage Instructions", "Lifestyle Recommendations" , "Additional Notes or Feedback" , "Prescribed Medication2" , "Dosage Instructions" , "Lifestyle Recommendations" , "Additional Notes or Feedback"] )
    # Select only required columns
    df = df[["Id", "Name", "Email", "Password"]]

    # display the DataFrame in a table format
    st.markdown("<h1 style='text-align: center; color: purple;'>List Of Registered Patients</h1>", unsafe_allow_html=True)
    # Add a search bar
    search = st.text_input("Search Patients:")

    for i in range(df.shape[0]):
        row = df.iloc[i]
        if search in row['Name'] or search in row['Email']:
            # Create a container for each patient
            with st.container():
                st.markdown("---")
                st.markdown(f"#### {row['Name']}")
                st.write(f"Email: {row['Email']}")
                st.write(f"Password: {row['Password']}")

                if st.button(f"Delete {row['Email']}"):
                    delete_user(row["Email"])
                    st.success(f"Deleted record for {row['Email']}!")




def manage_radiologist():
    # create a function to fetch all the users from the database
    def fetch_users():
        mycursor.execute("SELECT * FROM radiologist")
        users = mycursor.fetchall()
        return users

    # create a function to delete a user from the database
    def delete_user(email):
        mycursor.execute("DELETE FROM radiologist WHERE email = %s", (email,))
        mydb.commit()

    # fetch the list of users from the database
    users = fetch_users()

    # create a pandas DataFrame to hold the user data
    df = pd.DataFrame(users, columns=["Id" , "Name", "Email", "Password"])

    # add a "View" button and a "Delete" button to each row
    #df["View"] = df["Email"].apply(lambda email: st.button(f"View {email}"))


    # display the DataFrame in a table format
    st.markdown("<h1 style='text-align: center; color: purple;'>List Of Registered Doctors</h1>", unsafe_allow_html=True)
    st.dataframe(df)


def manage_doctors():
    # create a function to fetch all the users from the database
    def fetch_users():
        mycursor.execute("SELECT * FROM doctor")
        users = mycursor.fetchall()
        return users

    # create a function to delete a user from the database
    def delete_user(email):
        mycursor.execute("DELETE FROM doctor WHERE email = %s", (email,))
        mydb.commit()

    # fetch the list of users from the database
    users = fetch_users()

    # create a pandas DataFrame to hold the user data
    df = pd.DataFrame(users, columns=["Id", "Name", "Email", "Password"])

    # display the DataFrame in a table format
    st.markdown("<h1 style='text-align: center; color: purple;'>List Of Registered Doctors</h1>", unsafe_allow_html=True)

    # Add a search bar
    search = st.text_input("Search Doctors:")

    for i in range(df.shape[0]):
        row = df.iloc[i]
        if search in row['Name'] or search in row['Email']:
            # Create a container for each doctor
            with st.container():
                st.markdown("---")
                st.markdown(f"#### {row['Name']}")
                st.write(f"Email: {row['Email']}")
                st.write(f"Password: {row['Password']}")

                if st.button(f"Delete {row['Email']}"):
                    delete_user(row["Email"])
                    st.success(f"Deleted record for {row['Email']}!")

# def developer_information4():
#
#     def Introduction_Page():
#         st.title("Admin For Controlling All The Users")
#         #main_image = st.image('')
#         #readme_txt = st.markdown(get_file_content_as_string("/Introduction.md"))
#         st.write(""" But none of this should make you think poorly of TensorFlow; it remains an industry-proven library with support from one of the biggest companies on the
#         planet. PyTorch (backed, of course, by a different biggest company on the planet)""")
#         st.image("system_admin.jpg" , width = 700)
#
#
#     def About_Page():
#         st.title("About Page")
#         #main_image = st.image('')
#         #readme_txt = st.markdown(get_file_content_as_string("/About.md"))
#
#         st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.
#
#         """)
#
#         st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)
#
#
#
#     def Contact_Page():
#         st.title("Contact Page")
#         #main_image = st.image('')
#         #readme_txt = st.markdown(get_file_content_as_string("/Contact.md"))
#         st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.
#
#         """)
#
#         st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)
#
#
#
#     def Logout():
#         url = "http://localhost:8501/"
#         st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
#
#
#
#
#
#     selected = option_menu(
#      menu_title = "",
#      options = ["Home" , "About" , "Contact" , "Logout"] ,
#      orientation = "horizontal"
#     )
#
#
#     if selected == "Home":
#         # call the function home
#         Introduction_Page()
#     if selected == "About":
#         # call the About function
#         About_Page()
#
#     if selected == "Contact":
#         # call the contact page
#         Contact_Page()
#
#     if selected == "Logout":
#         # call the contact page
#         Logout()
#
#

#
# if login_option3 == "Home":
#     header_txt.empty()
#     Main_image.empty()
#     readme_text.empty()
#     developer_information4()

if login_option3 == "Manage User":
    # call the function for managing the user
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    manage_users()

if login_option3 == "Manage Radiologists":
    # call the function for managing the radiologists
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    manage_radiologist()

if login_option3 == "Manage Doctors":
    # call the function for managing the doctors
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    manage_doctors()
if login_option3 == "Logout":
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    url = "http://localhost:8501/"
    st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)








st.sidebar.title("Register Physicians")
login_option4 = st.sidebar.selectbox("Menu" , ("" , "Register Doctors" , "Register Radiologists" , "Logout"))
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")


def register_doctors():

    st.title("Registering New Doctors")
    # Define the form fields
    with st.form(key='doctor_form' , clear_on_submit=True):
        register_name = st.text_input("Name")
        register_email = st.text_input("Email")
        register_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button(label='Register')
        #reset_button = st.form_submit_button(label='Reset Form')

        if submit_button:
            if register_password == confirm_password:
                # Save the form data to the database
                mycursor.execute("INSERT INTO doctor (name, email, password) VALUES (%s, %s, %s)", (register_name, register_email, register_password))
                mydb.commit()
                st.success("Registration successful!")
                #st.write(f"Registered {register_name} successfully!")
            else:
                st.error("Passwords do not match.")


def register_radiologists():

    st.title("Registering New Radiologists")
    # Define the form fields
    with st.form(key='radiologist_form' , clear_on_submit=True):
        register_name = st.text_input("Name")
        register_email = st.text_input("Email")
        register_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button(label='Register')
        #reset_button = st.form_submit_button(label='Reset Form')

        if submit_button:
            if register_password == confirm_password:
                # Save the form data to the database
                mycursor.execute("INSERT INTO radiologist (name, email, password) VALUES (%s, %s, %s)", (register_name, register_email, register_password))
                mydb.commit()
                st.success("Registration successful!")
                #st.write(f"Registered {register_name} successfully!")
            else:
                st.error("Passwords do not match.")



#
# if login_option4 == "Home":
#     header_txt.empty()
#     Main_image.empty()
#     readme_text.empty()
#     developer_information4()

if login_option4 == "Register Doctors":
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    # call the function to register doctors
    register_doctors()

if login_option4 == "Register Radiologists":
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    # call the function to register radiologists
    register_radiologists()

if login_option4 == "Logout":
    header_txt.empty()
    Main_image.empty()
    #readme_text.empty()
    url = "http://localhost:8501/"
    st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)

#  ====================================== The Second Side Bar =======================================================================
st.sidebar.markdown("---")
st.sidebar.title(" Account Facilities ")
login_option2 = st.sidebar.selectbox("Configure Account" , ("" ,"Update Account" , "Delete Account"))
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")

def update_account():
    st.title("Update Your Account")
    # get the user's email and password to authenticate
    login_email = st.text_input("Email")
    login_password = st.text_input("Password", type="password")
    #check_button = st.button("Check")

    # check if the user exists in the database
    mycursor.execute("SELECT * FROM admin WHERE email = %s AND password = %s", (login_email, login_password))
    result = mycursor.fetchone()
    if result is not None:

        # display the user's old information and allow them to update it
        st.title("Update Your Old Informaton")
        #st.text_area(f"Name: {result[1]}")
        #st.text_area(f"Email: {result[2]}")

        # get the new information from the user
        new_name = st.text_input("New Name", value=result[1])
        new_email = st.text_input("New Email", value=result[2])
        new_password = st.text_input("New Password", type="password")

    # if the user submits the form, update the data in the database
        if st.button("Submit"):
            # update the user's data in the database
            mycursor.execute("UPDATE admin SET name = %s, email = %s, password = %s WHERE password = %s", (new_name, new_email, new_password, login_password))
            mydb.commit()

            # display a success message to the user
            st.success("Your information has been updated.")
            st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
    else:
          # display an error message if the user does not exist in the database
          st.error("Invalid email or password. Please try again.")


def delete_account():

    st.title("Delete Your Account")

    # get the user's email and password to authenticate
    login_email = st.text_input("Email")
    login_password = st.text_input("Password", type="password")
    #check_button = st.button("Check")

    # check if the user exists in the database
    mycursor.execute("SELECT * FROM admin WHERE email = %s AND password = %s", (login_email, login_password))
    result = mycursor.fetchone()
    # display the user's old information and allow them to update it
    st.title("Delete Your Informaton")

    # add a delete account button
    if st.button("Delete Account"):
        # delete the user's account from the database
        mycursor.execute("DELETE FROM admin WHERE password = %s", (login_password,))
        mydb.commit()

        # display a success message and exit the app
        st.success("Your account has been deleted.")
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
        st.stop()



if login_option2 == "Update Account":
    header_txt.empty()
    Main_image.empty()
    # call the function update account
    update_account()

if login_option2 == "Delete Account":
    header_txt.empty()
    Main_image.empty()
    # call the function delete account
    delete_account()
