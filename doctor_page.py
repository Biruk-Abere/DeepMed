import streamlit as st
import mysql.connector
from PIL import Image
from io import BytesIO
import os

# Create a connection to the MySQL database
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
header_txt = st.markdown(get_file_content_as_string('/instructions2.md'), unsafe_allow_html=True)
Main_image = st.image('The-structures-of-different-deep-learning-models.png', caption='Neural Networks Can Solve Almost Anything')
readme_text = st.markdown(get_file_content_as_string('/Instructions.md'), unsafe_allow_html=True)

st.sidebar.markdown("# Doctor's Feedback ")
doctors_dropdown = st.sidebar.selectbox("Give Feedback" , (" " , "Structural Machine Learning" , "Computer Vision" , "Chatbot") )
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")


if doctors_dropdown == "Structural Machine Learning":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()

    def display_patient_record():

        if 'search_done' not in st.session_state:
            st.session_state.search_done = False

        if 'patients' not in st.session_state:
            st.session_state.patients = []

        # Function to search for a patient by name
        def search_patient(name):
            query = "SELECT * FROM patients WHERE name LIKE %s"
            name = f"%{name}%"
            mycursor.execute(query, (name,))
            return mycursor.fetchall()

                # Physician's Page
        def physician_page():
            st.title("Physician's Page")
            st.write("Welcome, Physician!")

            # Search for a patient
            search_input = st.text_input("Search for a patient by name")
            if st.button("Search"):
                if search_input.strip():  # Check if the input field is not empty
                    st.session_state.search_done = True
                    patients = search_patient(search_input)
                    st.session_state.patients = patients
                    if patients:
                        st.success(f"Found {len(patients)} patient(s).")
                        st.markdown("---")
                    else:
                        st.warning("No patient found.")
                else:
                    st.warning("Please enter a patient name.")

        # Run the physician page
        physician_page()

        if st.session_state.search_done:
            for patient in st.session_state.patients:
                # Function to display the patient's medical record and update treatment plan
                st.subheader("Patient's Medical Record")
                st.markdown("---")

                patient_info = {
                    "Patient ID": patient[0],
                    "Name": patient[1],
                    "Email": patient[2],
                    "Gender": patient[4],
                    "Age": patient[5],
                    "Symptoms": patient[6],
                    "prediction": patient[7]
                }

                for key, value in patient_info.items():
                    st.write(f"**{key}:** {value}")

                st.markdown("---")
                st.subheader("Doctor's Treatment Plan If Any")

                with st.form("treatment_plan_form" , clear_on_submit = True):
                    medication = st.text_input("Prescribed Medication")
                    dosage = st.text_input("Dosage Instructions")
                    lifestyle_recommendations = st.text_area("Lifestyle Recommendations")
                    additional_notes = st.text_area("Additional Notes or Feedback")

                    submit_button = st.form_submit_button("Submit Treatment Plan")

                # Handle treatment plan submission
                if submit_button:
                    # Update the treatment plan in the database for the patient
                    update_query = "UPDATE patients SET medication2 = %s, dosage2 = %s, lifestyle_recommendations2 = %s, additional_notes2 = %s WHERE id = %s"
                    mycursor.execute(update_query, (medication, dosage, lifestyle_recommendations, additional_notes, patient[0]))
                    mydb.commit()

                    # Send notification to the patient
                    st.success("Treatment plan submitted successfully. Notification sent to the patient.")

    display_patient_record()


if doctors_dropdown == "Computer Vision":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()

    def display_patient_record():
        # Create a connection to the MySQL database
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="deeplearning"
        )

        # Create a cursor object
        mycursor = mydb.cursor()

        st.title("Physician's Page")
        st.write("Welcome, Physician!")

        # Search for a patient
        patient_name = st.text_input('Enter the name of the patient:', value=st.session_state.get('patient_name', ''))
        if st.button("Search"):
            if patient_name.strip():  # Check if the input field is not empty
                st.session_state['patient_name'] = patient_name
                mycursor.execute("SELECT * FROM patients WHERE name = %s", (patient_name,))
                record = mycursor.fetchone()
                if record:  # Check if a patient is found
                    st.session_state['record'] = record
                else:  # If no patient is found
                    st.warning("No patient found!")
            else:  # If the input field is empty
                st.warning("Please enter a patient name.")


        if 'record' in st.session_state:
            record = st.session_state['record']
            st.markdown("---")
            # Display patient's record
            st.write("Patient's Record:")
            st.write("Name: ", record[1])  # Assuming 'name' is the second column in your table
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                # Display patient's medical image
                st.write("### Medical Image")
                st.write("-" * 40)
                image_bytes = record[10]  # Assuming 'image' is the third column from the end
                image = Image.open(BytesIO(image_bytes))
                st.image(image, width=180)

            with col2:
                # Display Category
                st.write("### Category")
                st.write("-" * 40)
                st.write(record[11])  # Assuming 'category' is the second column from the end

            with col3:
                # Display Confidence
                st.write("### Confidence")
                st.write("-" * 40)
                st.write(record[12])  # Assuming 'confidence' is the last column

            st.markdown("---")

            # Physician reviews the prediction
            st.subheader("Doctor's Treatment Plan If Any")

            with st.form("treatment_plan_form" , clear_on_submit = True):
                medication = st.text_input("Prescribed Medication", value=st.session_state.get('medication', ''))
                dosage = st.text_input("Dosage Instructions", value=st.session_state.get('dosage', ''))
                lifestyle_recommendations = st.text_area("Lifestyle Recommendations", value=st.session_state.get('lifestyle_recommendations', ''))
                additional_notes = st.text_area("Additional Notes or Feedback", value=st.session_state.get('additional_notes', ''))

                if st.form_submit_button("Submit Treatment Plan"):
                    # SQL query to update the patient's treatment plan
                    sql = "UPDATE patients SET medication = %s, dosage = %s, lifestyle_recommendations = %s, additional_notes = %s WHERE name = %s"
                    val = (medication, dosage, lifestyle_recommendations, additional_notes, patient_name)
                    mycursor.execute(sql, val)
                    mydb.commit()

                    # Clear the form input after submit
                    st.session_state['medication'] = ''
                    st.session_state['dosage'] = ''
                    st.session_state['lifestyle_recommendations'] = ''
                    st.session_state['additional_notes'] = ''

                    st.success("Treatment Plan Sent To The Patient Successfully")



    display_patient_record()


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
    mycursor.execute("SELECT * FROM doctor WHERE email = %s AND password = %s", (login_email, login_password))
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
            mycursor.execute("UPDATE doctor SET name = %s, email = %s, password = %s WHERE password = %s", (new_name, new_email, new_password, login_password))
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
    mycursor.execute("SELECT * FROM doctor WHERE email = %s AND password = %s", (login_email, login_password))
    result = mycursor.fetchone()
    # display the user's old information and allow them to update it
    st.title("Delete Your Informaton")

    # add a delete account button
    if st.button("Delete Account"):
        # delete the user's account from the database
        mycursor.execute("DELETE FROM doctor WHERE password = %s", (login_password,))
        mydb.commit()

        # display a success message and exit the app
        st.success("Your account has been deleted.")
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
        st.stop()


if login_option2 == "Update Account":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()
    # call the function update account
    update_account()

if login_option2 == "Delete Account":
    header_txt.empty()
    Main_image.empty()
    readme_text.empty()
    # call the function delete account
    delete_account()
