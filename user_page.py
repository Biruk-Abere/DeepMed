import streamlit as st
from PIL import Image
from tensorflow.keras.models import model_from_json
from streamlit_option_menu import option_menu
import requests
import random
import pickle
import numpy as np
import io
import os
import torch
import torchvision
from torchvision import models, transforms
import json
import random
from model import NeuralNet
from preprocess import bag_of_words , tokenize
import re
import mysql.connector
from PIL import Image
from io import BytesIO
import pandas as pd
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from PIL import ImageOps


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

# ======================================= Introduction Home Page ==============================================

def developer_information():

    def Introduction_Page():
        st.title("Medical Image Classification & Diagnosis")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Introduction.md"))
        st.write(""" But none of this should make you think poorly of TensorFlow; it remains an industry-proven library with support from one of the biggest companies on the
        planet. PyTorch (backed, of course, by a different biggest company on the planet) is, I
        would say, a more streamlined and focused approach to deep learning and differential
        programming. Because it doesn’t have to continue supporting older, crustier APIs, it
        is easier to teach and become productive in PyTorch than in TensorFlow.""")
        st.image("computer_vision.png")


    def About_Page():
        st.title("About Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/About.md"))

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Contact_Page():
        st.title("Contact Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Contact.md"))
        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Logout():
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)





    selected = option_menu(
     menu_title = "",
     options = ["Home" , "About" , "Contact" , "Logout"] ,
     orientation = "horizontal"
    )


    if selected == "Home":
        # call the function home
        Introduction_Page()
    if selected == "About":
        # call the About function
        About_Page()

    if selected == "Contact":
        # call the contact page
        Contact_Page()

    if selected == "Logout":
        # call the contact page
        Logout()



def developer_information2():

    def Introduction_Page2():
        st.title("Generative Question Answering Using GPT-3")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Introduction.md"))
        st.write(""" GPT (Generative Pre-trained Transformer) is a deep learning model introduced by OpenAI in 2018. GPT is based on the Transformer architecture, which is a neural network architecture that is particularly suited for natural language processing (NLP) tasks such as language modeling, machine translation, and question-answering.

The Transformer architecture consists of two main components: the encoder and the decoder. The encoder takes an input sequence of tokens and generates a sequence of encoded vectors, while the decoder takes the encoded vectors and generates a sequence of output tokens. GPT, in particular, is an autoregressive language model based on the decoder-only Transformer architecture.""")
        st.image("transformer.png" , width = 700)


    def About_Page2():
        st.title("About Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/About.md"))

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Contact_Page2():
        st.title("Contact Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Contact.md"))
        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Logout():
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)





    selected = option_menu(
     menu_title = "",
     options = ["Home" , "About" , "Contact" , "Logout"] ,
     orientation = "horizontal"
    )


    if selected == "Home":
        header_txt.empty()
        Main_image.empty()

        # call the function home
        Introduction_Page2()
    if selected == "About":
        # call the About function
        About_Page2()

    if selected == "Contact":
        # call the contact page
        Contact_Page2()

    if selected == "Source Code":
        # call the contact page
        source_code2()


def developer_information3():

    def Introduction_Page3():
        st.title("Structured Machine Learning ")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Introduction.md"))
        st.write(""" GPT (Generative Pre-trained Transformer) is a deep learning model introduced by OpenAI in 2018. GPT is based on the Transformer architecture, which is a neural network architecture that is particularly suited for natural language processing (NLP) tasks such as language modeling, machine translation, and question-answering.

The Transformer architecture consists of two main components: the encoder and the decoder. The encoder takes an input sequence of tokens and generates a sequence of encoded vectors, while the decoder takes the encoded vectors and generates a sequence of output tokens. GPT, in particular, is an autoregressive language model based on the decoder-only Transformer architecture.""")
        st.image("The-structures-of-different-deep-learning-models.png")


    def About_Page3():
        st.title("About Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/About.md"))

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Contact_Page3():
        st.title("Contact Page")
        #main_image = st.image('')
        #readme_txt = st.markdown(get_file_content_as_string("/Contact.md"))
        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation.

        """)

        st.write(""" In the first sample essay from mechanical engineering, what stands out immediately are the length and the photographs. In this case, the student was applying for an engineering scholarship, so he was given room to flesh out technical material as well as address issues such as personal motivations one would expect to read in a personal statement. Much of the essay is given to a discussion of his thesis work, which involves the examination of “the propagation of a flame in a small glass tube.” The figures depict the experimental work and represent the success of preliminary thesis results, visually indicating the likely point at which the flame reached detonation. """)



    def Logout():
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)





    selected = option_menu(
     menu_title = "",
     options = ["Home" , "About" , "Contact" , "Logout"] ,
     orientation = "horizontal"
    )


    if selected == "Home":
        # call the function home
        Introduction_Page3()
    if selected == "About":
        # call the About function
        About_Page3()

    if selected == "Contact":
        # call the contact page
        Contact_Page3()

    if selected == "Logout":
        # call the contact page
        Logout()



# =====================    This is for the side menu for selecting the sections of the app ===================
st.sidebar.markdown("# Computer Vision")
option = st.sidebar.selectbox("Choose Disease" , (" " , "Introduction Page" ,  "Covid-19" , "Melanoma" , "Diabetes_Retinopathy" , "Vision Feedback"))
nlp_empty = st.sidebar.empty()  # create empty space below selectbox
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")

st.sidebar.markdown("")
st.sidebar.markdown("---")

# =========================  Condition if the user chooses Natural Language processing  ========================================================
if option == "Introduction Page":
    header_txt.empty()
    Main_image.empty()

    # alert options for further instructions to proceed
    #success_text = st.sidebar.success("To Continue 'Choose Disease' ")
    #warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")

    # display the developer information
    developer_information()
if option == "Covid-19":
    header_txt.empty()
    Main_image.empty()

    # Load your trained model
    model = tf.keras.models.load_model('best-model-weighted.h5')
    st.title("COVID-19 Image Classifier")

    password = st.text_input('Please enter your password', type='password')

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.button("Login"):
        mycursor.execute("SELECT * FROM patients WHERE password = %s", (password,))
        record = mycursor.fetchone()
        if record:
            st.session_state["logged_in"] = True
        else:
            st.error('The provided password is incorrect.')

    if st.session_state["logged_in"]:
        # Use a list to store uploaded files
        uploaded_files = st.file_uploader(" ", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files:
            # Display total number of uploaded images
            st.write("-" * 34)
            st.success(f"Total Medical Images Categorized: {len(uploaded_files)}")
            st.write("-" * 34)

            class_names = ['COVID', 'NORMAL']  # classes used for your model

            for uploaded_file in uploaded_files:
                # Process the image
                image = Image.open(uploaded_file).convert("RGB")
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0

                # Make a prediction
                predictions = model.predict(img_array_expanded)

                # Write static sections for each image
                col1, col2, col3 = st.columns([1.5, 2, 1])

                with col1:
                    st.image(img, width=180)
                with col2:
                    st.write("### Medical Category")
                    for class_name, prediction in zip(class_names, predictions[0]):
                        st.write(f"* {class_name}")
                with col3:
                    st.write("### Confidence")
                    for class_name, prediction in zip(class_names, predictions[0]):
                        st.write(f"* {prediction*100:.2f}%")

                # Store the image, category and confidence in the database
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                sql = "UPDATE patients SET image = %s, category = %s, confidence = %s WHERE password = %s"
                val = (img_byte_arr, class_names[np.argmax(predictions)], float(np.max(predictions)), password)
                mycursor.execute(sql, val)
                mydb.commit()

                # Separator
                st.write("-" * 80)

if option == "Vision Feedback":

    header_txt.empty()
    Main_image.empty()

    st.markdown("<h1 style='text-align: center; color: purple;'>Here is your Doctor's Feedback</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: black;'>Welcome, Patient!</div>", unsafe_allow_html=True)


    # Create a connection to the MySQL database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="deeplearning"
    )

    # Create a cursor object
    mycursor = mydb.cursor()

    def fetch_patient_record(patient_password):
        # Query the database for the patient's record
        query = "SELECT * FROM patients WHERE password = %s"
        mycursor.execute(query, (patient_password,))
        return mycursor.fetchone()

    def display_patient_record(record):
        # Display patient information
        st.markdown("---")
        st.markdown("<h2 style='color: blue;'>Your Medical Record</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"<div style='color: black;'>Name: | <span style='margin-left: 20px;'>{record[1]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Email:| <span style='margin-left: 20px;'> {record[2]}</span></div>", unsafe_allow_html=True)  # Assuming 'email' is the third column in your table
        st.markdown(f"<div style='color: black;'>Gender:|<span style='margin-left: 20px;'>{record[4]}</span></div>", unsafe_allow_html=True)  # Assuming 'gender' is the fifth column in your table
        st.markdown(f"<div style='color: black;'>Age:|| <span style='margin-left: 20px;'> {record[5]}</span></div>", unsafe_allow_html=True)  # Assuming 'age' is the sixth column in your table
        st.markdown("---")

        # Display the treatment plan
        st.markdown("<h2 style='color: green;'>Your Treatment Plan</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"<div style='color: black;'>Medication: | <span style='margin-left: 20px;'> {record[13]}</span></div>", unsafe_allow_html=True)  # Assuming 'medication' is the ninth column in your table
        st.markdown(f"<div style='color: black;'>Dosage: | <span style='margin-left: 20px;'> {record[14]}</span></div>", unsafe_allow_html=True)  # Assuming 'dosage' is the tenth column in your table
        st.markdown(f"<div style='color: black;'>Lifestyle Recommendations: | <span style='margin-left: 20px;'> {record[15]} </span></div>", unsafe_allow_html=True)  # Assuming 'lifestyle_recommendations' is the eleventh column in your table
        st.markdown(f"<div style='color: black;'>Additional Notes: | <span style='margin-left: 20px;'> {record[16]}</span></div>", unsafe_allow_html=True)  # Assuming 'additional_notes' is the twelfth column in your table

    # Allow patient to input their name
    patient_password = st.text_input("Enter your password" , type = "password")
    if st.button("Fetch Record"):
        record = fetch_patient_record(patient_password)
        if record:
            display_patient_record(record)
        else:
            st.warning("No patient found with that password.")

# ===================================================================================================================

# =================================  Define The ChatBot Function ==================================================
def chatbot():
    with open("intents.json" , "r") as f:
        intents = json.load(f)

    FILE = "data_chatbot.pth"
    data =  torch.load(FILE)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    #model_state = data["model_state"]

    model = NeuralNet(input_size , hidden_size , output_size)
    if "chatbot_model" not in st.session_state:
        # display the app status on the screen
        status_1 = st.markdown("Loading The Model")
        # get the model and the weights
        model.load_state_dict(torch.load("chatbot-model-weighted.pt" , torch.device('cpu')) )
        # close the app status displayed above
        status_1.empty()
    #

        # call the streamlit app function



        # Define chatbot function
        def chatbot_response(sentence):
            # Tokenize sentence
            tokenized_sentence = tokenize(sentence)
            # Convert tokenized sentence to bag of words
            sentence_bag = bag_of_words(tokenized_sentence, all_words)
            # Reshape sentence bag
            sentence_bag = sentence_bag.reshape(1, sentence_bag.shape[0])
            # Convert sentence bag to torch tensor
            sentence_tensor = torch.from_numpy(sentence_bag)
            # Make prediction with model
            output = model(sentence_tensor)
            # Get index of highest probability
            index = output.argmax().item()
            # Get tag from index
            tag = tags[index]
            probs = torch.softmax(output, dim=1)
            prob = probs[0][index]

            if prob.item() > 0.75:
                # Get list of all intents
                all_intents = intents['intents']
                # Find intent with matching tag
                for intent in all_intents:
                    if intent['tag'] == tag:
                        # Get list of patterns for intent
                        patterns = intent['patterns']
                        # Find matching pattern
                        for pattern in patterns:
                            # Check if sentence matches pattern, ignoring case and similar words
                            if re.search(pattern, sentence, re.IGNORECASE):
                                # Get list of responses for intent
                                responses = intent['responses']
                                # Return response in order of patterns
                                for i, p in enumerate(patterns):
                                    if re.search(p, sentence, re.IGNORECASE):
                                        return responses[i]
                # If no matching pattern found, return random response
                responses = intent['responses']
                return responses
                #return random.choice(responses)
            else:
                string = "Sorry I do not understand!"
                return string


            # Define Streamlit app
        def app():
            # Set title and page icon
            # st.set_page_config(page_title='Chatbot', page_icon=':speech_balloon:')
            # Set app title
            st.title('GPT Question Answering Chatbot')

            # Set input field for user question
            user_input = st.text_input('Ask a question')
            # Check if user has asked a question
            if st.button("Answer") or user_input:
                # Get chatbot response
                chatbot_output = chatbot_response(user_input)
                # Display chatbot response in chat bubble
                st.text_area('Chatbot', chatbot_output, height=10)

        # call the chat app
        app()






st.sidebar.markdown("# Natural Language Processing")
nlp_option = st.sidebar.selectbox("Chat With Me" , (" " , "Explanation Page", "Generative ChatBot" , "Doctor's Feedback"))
nlp_empty = st.sidebar.empty()  # create empty space below selectbox
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")

if nlp_option == "Explanation Page":

    header_txt.empty()
    Main_image.empty()

    developer_information2()
elif nlp_option == "Generative ChatBot":

    header_txt.empty()
    Main_image.empty()

    # call the chatbot function
    chatbot()
    # # display chatbot content
    # st.title('GPT Question Answering')
    # question = st.text_area('Enter your question:', '')
    # if st.button('Generate Answer'):
    #     # call the chatbot function
    #     chatbot()



#  ============================================   STRUCTURED MACHINE LEARNING ===============================================================
st.sidebar.markdown("---")

st.sidebar.markdown("# Structured Machine Learning")
structured_machine_learning = st.sidebar.selectbox("Medical Prediction" , (" " , "Description Page", "Covid-19" , "Diabetes" , "Heart Disease" , "Medical Prognosis" , "ML Feedback") )
success_text = st.sidebar.success("To Continue 'Choose Disease' ")
warning_text = st.sidebar.warning("To Chat Select 'Chat With Me'")



if structured_machine_learning == "Description Page":

    header_txt.empty()
    Main_image.empty()

    developer_information3()


#  =======================================    Covid-19 Prediction ====================================

#loading the Covid-19 dataset
df1=pd.read_csv("Covid-19 Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x1) & target(y1)
x1=df1.drop("Infected with Covid19",axis=1)
x1=np.array(x1)
y1=pd.DataFrame(df1["Infected with Covid19"])
y1=np.array(y1)
#performing train-test split on the data
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model1=RandomForestClassifier()
#fitting the model with train data (x1_train & y1_train)
model1.fit(x1_train,y1_train)

#heading over to the Covid-19 section
if structured_machine_learning=="Covid-19":

    header_txt.empty()
    Main_image.empty()

    st.title("Know If You Are Affected By Covid")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Dry Cough (drycough), Fever (fever), Sore Throat (sorethroat), Breathing Problem (breathingprob)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]

    #prediction part predicts whether the person is affected by Covid-19 or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):

        if prediction1=="Yes":
            st.warning("You Might Be Affected By Covid-19")
            st.markdown("---")
            st.header(" Don't Panic Here is What You Need To Do")
            st.success(""" First and foremost, it is crucial to isolate yourself and avoid contact with others to prevent the virus's spread. Quarantine yourself for at least ten days and monitor your symptoms closely. If your symptoms worsen, seek medical attention immediately.

Stay hydrated by drinking plenty of fluids and rest as much as possible. This will help your body fight off the infection and recover more quickly. Over-the-counter pain relievers like acetaminophen or ibuprofen can help alleviate fever, aches, and pains. Avoid smoking and secondhand smoke, which can damage your lungs, and exacerbate respiratory symptoms.

Make sure to follow any instructions from your healthcare provider regarding medications, monitoring, and follow-up care. Your doctor may prescribe medication to help alleviate your symptoms, or recommend certain treatments, like oxygen therapy, if necessary.

In addition to taking care of yourself, it is important to prevent the spread of the virus to others. Wear a mask whenever you are around others, and practice good hand hygiene by washing your hands frequently and thoroughly. Cover your mouth and nose when you cough or sneeze and dispose of used tissues properly.

Finally, be sure to stay informed about the latest developments in the pandemic and follow guidelines from public health officials. Keep up-to-date with any changes in your local area and any new treatments or preventative measures that may become available.
            """)

        elif prediction1=="No":
            st.success("You Are Safe")
            st.markdown("---")
            st.header("Do The Following For Future Care")
            st.warning("It is important to take steps to protect yourself and others from potential exposure. This includes following guidelines set by health organizations and governments, such as practicing good hand hygiene, wearing a mask in public settings, and maintaining physical distance from others. It is also important to stay informed about any updates or changes in guidelines, as the situation surrounding the pandemic is constantly evolving. Additionally, taking care of one's overall health and immune system through regular exercise, a balanced diet, and adequate sleep can help reduce the risk of contracting any illnesses in the future.")





# =========================== Diabetes Prediction ====================================

#loading the Diabetes dataset
df2=pd.read_csv("Diabetes Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
#performing train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)



#heading over to the Diabetes section
if structured_machine_learning == "Diabetes":

    header_txt.empty()
    Main_image.empty()


    st.title("Know If You Are Affected By Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]

    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You Might Be Affected By Diabetes")
            st.markdown("---")
            st.header(" Don't Panic Here is What You Need To Do")
            st.success(""" First and foremost, it is crucial to isolate yourself and avoid contact with others to prevent the virus's spread. Quarantine yourself for at least ten days and monitor your symptoms closely. If your symptoms worsen, seek medical attention immediately.

Stay hydrated by drinking plenty of fluids and rest as much as possible. This will help your body fight off the infection and recover more quickly. Over-the-counter pain relievers like acetaminophen or ibuprofen can help alleviate fever, aches, and pains. Avoid smoking and secondhand smoke, which can damage your lungs, and exacerbate respiratory symptoms.

Make sure to follow any instructions from your healthcare provider regarding medications, monitoring, and follow-up care. Your doctor may prescribe medication to help alleviate your symptoms, or recommend certain treatments, like oxygen therapy, if necessary.

In addition to taking care of yourself, it is important to prevent the spread of the virus to others. Wear a mask whenever you are around others, and practice good hand hygiene by washing your hands frequently and thoroughly. Cover your mouth and nose when you cough or sneeze and dispose of used tissues properly.

Finally, be sure to stay informed about the latest developments in the pandemic and follow guidelines from public health officials. Keep up-to-date with any changes in your local area and any new treatments or preventative measures that may become available.
            """)

        elif prediction2==0:
            st.success("You Are Safe")
            st.markdown("---")
            st.header("Do The Following For Future Care")
            st.warning("It is important to take steps to protect yourself and others from potential exposure. This includes following guidelines set by health organizations and governments, such as practicing good hand hygiene, wearing a mask in public settings, and maintaining physical distance from others. It is also important to stay informed about any updates or changes in guidelines, as the situation surrounding the pandemic is constantly evolving. Additionally, taking care of one's overall health and immune system through regular exercise, a balanced diet, and adequate sleep can help reduce the risk of contracting any illnesses in the future.")


#  ========================================= Heart Disease Prediction =========================================================

#loading the Heart Disease dataset
df3=pd.read_csv("Heart Disease Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
x3=df3.iloc[:,[2,3,4,7]].values
x3=np.array(x3)
y3=y3=df3.iloc[:,[-1]].values
y3=np.array(y3)
#performing train-test split on the data
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model3=RandomForestClassifier()
#fitting the model with train data (x3_train & y3_train)
model3.fit(x3_train,y3_train)


#heading over to the Heart Disease section
if structured_machine_learning=="Heart Disease":

    header_txt.empty()
    Main_image.empty()


    st.title("Know If You Are Affected By Heart Disease")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Chest Pain (chestpain), Blood Pressure-BP (bp), Cholestrol (cholestrol), Maximum HR (maxhr)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    chestpain=st.number_input("Rate Your Chest Pain (1-4)",min_value=1,max_value=4,step=1)
    bp=st.number_input("Enter Your Blood Pressure Rate (95-200)",min_value=95,max_value=200,step=1)
    cholestrol=st.number_input("Enter Your Cholestrol Level Value (125-565)",min_value=125,max_value=565,step=1)
    maxhr=st.number_input("Enter You Maximum Heart Rate (70-200)",min_value=70,max_value=200,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction3=model3.predict([[chestpain,bp,cholestrol,maxhr]])[0]

    #prediction part predicts whether the person is affected by Heart Disease or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if str(prediction3)=="Presence":
            st.warning("You Might Be Have a Heart Disease")
            st.markdown("---")
            st.header(" Don't Panic Here is What You Need To Do")
            st.success(""" First and foremost, it is crucial to isolate yourself and avoid contact with others to prevent the virus's spread. Quarantine yourself for at least ten days and monitor your symptoms closely. If your symptoms worsen, seek medical attention immediately.

Stay hydrated by drinking plenty of fluids and rest as much as possible. This will help your body fight off the infection and recover more quickly. Over-the-counter pain relievers like acetaminophen or ibuprofen can help alleviate fever, aches, and pains. Avoid smoking and secondhand smoke, which can damage your lungs, and exacerbate respiratory symptoms.

Make sure to follow any instructions from your healthcare provider regarding medications, monitoring, and follow-up care. Your doctor may prescribe medication to help alleviate your symptoms, or recommend certain treatments, like oxygen therapy, if necessary.

In addition to taking care of yourself, it is important to prevent the spread of the virus to others. Wear a mask whenever you are around others, and practice good hand hygiene by washing your hands frequently and thoroughly. Cover your mouth and nose when you cough or sneeze and dispose of used tissues properly.

Finally, be sure to stay informed about the latest developments in the pandemic and follow guidelines from public health officials. Keep up-to-date with any changes in your local area and any new treatments or preventative measures that may become available.
            """)

        elif str(prediction3)=="Absence":
            st.success("You Are Safe")
            st.header("Do The Following For Future Care")
            st.warning("It is important to take steps to protect yourself and others from potential exposure. This includes following guidelines set by health organizations and governments, such as practicing good hand hygiene, wearing a mask in public settings, and maintaining physical distance from others. It is also important to stay informed about any updates or changes in guidelines, as the situation surrounding the pandemic is constantly evolving. Additionally, taking care of one's overall health and immune system through regular exercise, a balanced diet, and adequate sleep can help reduce the risk of contracting any illnesses in the future.")



if structured_machine_learning == "Medical Prognosis":
    header_txt.empty()
    Main_image.empty()

    st.title("Medical Prognosis From The Symptoms")
    # Create a connection to the MySQL database
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="",
      database="deeplearning"
    )

    # Create a cursor object
    mycursor = mydb.cursor()

    # Check if the patients table already exists
    mycursor.execute("SHOW TABLES LIKE 'patients'")
    result = mycursor.fetchone()

    if result is None:
        mycursor.execute('''CREATE TABLE patients
                     (id INT AUTO_INCREMENT PRIMARY KEY,
                      name VARCHAR(255),
                      email VARCHAR(255),
                      password VARCHAR(255),
                      gender VARCHAR(255),
                      age INT,
                      symptoms TEXT,
                      prediction VARCHAR(255),
                      feedback TEXT,
                      treatment_plan TEXT)''')




    # Define the symptoms as a list
    symptoms = ["itching", "skin_rash", "continuous_sneezing", "joint_pain",
                "stomach_pain", "acidity", "vomiting", "fatigue", "anxiety",
                "weight_loss", "restlessness", "cough", "high_fever", "breathlessness",
                "sweating", "dehydration", "indigestion", "headache", "dark_urine",
                "nausea", "loss_of_appetite", "back_pain", "abdominal_pain",
                "diarrhoea", "mild_fever", "yellow_urine", "chest_pain",
                "fast_heart_rate", "neck_pain", "obesity", "knee_pain",
                "loss_of_balance", "loss_of_smell", "depression", "muscle_pain",
                "belly_pain", "lack_of_concentration", "visual_disturbances",
                "coma", "stomach_bleeding"]


    patient_name = st.text_input("Enter Your Name")
    patient_gender = st.selectbox("Gender", ("Male", "Female"))
    patient_age = st.slider("Age", 0, 100, 5)

    # Create the multiselect widget for symptoms
    selected_symptoms = st.multiselect("Select your symptoms", symptoms, key="selected_symptoms")

    # Set the selected symptoms as True in the dictionary
    symptoms_dict = dict.fromkeys(symptoms, False)
    for symptom in selected_symptoms:
        symptoms_dict[symptom] = True

    # Create a comma-separated string of the selected symptoms
    symptoms_str = ", ".join(selected_symptoms)

    # Load the pre-trained Scikit-Learn model
    with open("disease_prediction_model.pkl", "rb") as file:
        data = pickle.load(file)
    model = data["model"]

    # Create a button to predict the disease based on the selected symptoms
    if st.button("Predict Disease"):
        # Query the database to find the corresponding row for the entered name
        mycursor.execute("SELECT * FROM patients WHERE name = %s", (patient_name,))
        patient = mycursor.fetchone()

        if patient:
            # Create a dataframe with the selected symptoms
            symptoms_df = pd.DataFrame([symptoms_dict])

            # Use the pre-trained model to predict the disease
            prediction = model.predict(symptoms_df)

            # Display the prediction
            st.success(f"Based on your symptoms, you may have {prediction[0]}")

            # Update the patient's row with the new symptom information
            update_query = "UPDATE patients SET age = %s, gender = %s, symptoms = %s, prediction = %s WHERE name = %s"

            # Execute the update query with the new values
            mycursor.execute(
                update_query, (patient_age, patient_gender, symptoms_str, prediction[0], patient_name)
            )
            mydb.commit()
        else:
            st.error("Log In to access AI models")

if structured_machine_learning == "ML Feedback":
    header_txt.empty()
    Main_image.empty()

    st.markdown("<h1 style='text-align: center; color: purple;'>Here is your Doctor's Feedback</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: black;'>Welcome, Patient!</div>", unsafe_allow_html=True)

    # Function to fetch patient record by name
    def fetch_patient(password):
        query = "SELECT * FROM patients WHERE password = %s"
        mycursor.execute(query, (password,))
        return mycursor.fetchone()

    # Function to display patient's medical record and doctor's feedback
    def display_patient_record(patient):

        st.markdown("---")
        st.markdown("<h2 style='color: red;'>Your Medical Record</h2>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown(f"<div style='color: black;'>Name: | <span style='margin-left: 20px;'>{patient[1]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Email: | <span style='margin-left: 20px;'>{patient[2]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Gender: | <span style='margin-left: 20px;'>{patient[4]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Age: | <span style='margin-left: 20px;'>{patient[5]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Symptoms: | <span style='margin-left: 20px;'>{patient[6]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Prediction: | <span style='margin-left: 20px;'>{patient[7]}</span></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2 style='color: green;'>Your Treatment Plan</h2>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown(f"<div style='color: black;'>Medication: | <span style='margin-left: 20px;'>{patient[-4]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Dosage: | <span style='margin-left: 20px;'>{patient[-3]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Lifestyle Recommendations: | <span style='margin-left: 20px;'>{patient[-2]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: black;'>Additional Notes: | <span style='margin-left: 20px;'>{patient[-1]}</span></div>", unsafe_allow_html=True)

    patient_password = st.text_input("Enter your password:", type="password")
    if st.button("Fetch Record"):
        patient = fetch_patient(patient_password)
        if patient:
            display_patient_record(patient)
        else:
            st.error("No record found for the entered name.")


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
    mycursor.execute("SELECT * FROM patients WHERE email = %s AND password = %s", (login_email, login_password))
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
            mycursor.execute("UPDATE patients SET name = %s, email = %s, password = %s WHERE password = %s", (new_name, new_email, new_password, login_password))
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
    mycursor.execute("SELECT * FROM patients WHERE email = %s AND password = %s", (login_email, login_password))
    result = mycursor.fetchone()
    # display the user's old information and allow them to update it
    st.title("Delete Your Informaton")

    # add a delete account button
    if st.button("Delete Account"):
        # delete the user's account from the database
        mycursor.execute("DELETE FROM patients WHERE password = %s", (login_password,))
        mydb.commit()

        # display a success message and exit the app
        st.success("Your account has been deleted.")
        url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)
        st.stop()


if login_option2 == "Update Account":
    # call the function update account
    header_txt.empty()
    Main_image.empty()

    update_account()

if login_option2 == "Delete Account":
    # call the function delete account
    header_txt.empty()
    Main_image.empty()

    delete_account()
