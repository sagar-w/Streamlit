import streamlit as st
#import plotly.express as px
import snowflake.connector
import pandas as pd
import time
from st_aggrid import AgGrid
from st_aggrid import  GridOptionsBuilder, DataReturnMode, GridUpdateMode
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

# set page width
st.set_page_config(layout="wide", page_title='ML MODEL ON SNOWFLAKE', page_icon='🎡')



# set page background
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://i.gifer.com/70bm.gi");
        background-attachment: fixed;        
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        background-color: white;
        color: black;

         }}
         </style>
         """,
        unsafe_allow_html=True
    )
add_bg_from_url()


# Connection to Snowflake Database
ctx = snowflake.connector.connect(
    user="m*****amlit",
    password="Ml***********123",
    account="SJ2****.ap-south-1.aws"
)
ctx.cursor().execute('USE warehouse COMPUTE_WH')
ctx.cursor().execute('USE database HEART_DB')
ctx.cursor().execute('USE schema PUBLIC')


# Title and current date display
st.markdown("<h1 style='text-align: center; color:BLACK;'>Cardiovascular Disease Prediction</h1>", unsafe_allow_html=True)

# Define the function to display the home page
def home():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write('To properly forecast heart health state, a machine learning model was developed using Snowpark, cardiovascular information stored in Snowflake. Deployed on Streamlit for a user-friendly interaction. The approach can help healthcare professionals identify probable heart health problems early, allowing for immediate attention and better patient outcomes.')
    #st.title("Cardiovascular Disease Data Exploration")
    st.markdown("<h5 style='text-align: center; color:#0020C2;'>Cardiovascular Disease Dataset Exploration</h5>", unsafe_allow_html=True)

    query = f"SELECT  * from PREDICTED_DATA"
    cur = ctx.cursor().execute(query)
    data_df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
    # Show some basic information about the dataset
    df = data_df

    data = [
        ['AGE', 'Age', 'The age of the patient'],
        ['SEX', 'Sex', 'The gender of the patient'],
        ['CP', 'Chest Pain Type', 'The type of chest pain experienced by the patient'],
        ['TRESTBPS', 'Resting Blood Pressure', "The patient's resting blood pressure (in mm Hg)"],
        ['CHOL', 'Serum Cholesterol', "The patient's serum cholesterol level (in mg/dl)"],
        ['FBS', 'Fasting Blood Sugar', "The patient's fasting blood sugar level (in mg/dl)"],
        ['RESTECG', 'Resting Electrocardiographic Results', "The results of the patient's resting electrocardiogram"],
        ['THALACH', 'Maximum Heart Rate Achieved', "The patient's maximum heart rate achieved during exercise"],
        ['EXANG', 'Exercise Induced Angina', 'Whether or not the patient experienced angina during exercise'],
        ['OLDPEAK', 'ST Depression Induced by Exercise', 'The amount of ST depression induced by exercise relative to rest'],
        ['THAL', 'Thalassemia', 'The type of thalassemia the patient has']
    ]

    abr_df = pd.DataFrame(data, columns=['Feature', 'Full Form', 'Description'])

    st.table(abr_df)
    
    g1, g2 = st.columns((4, 4))
    with g1:
        st.write("Correlation Matrix")
        corr_matrix = df.corr()
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
        plt.title('Correlation Matrix')
        st.pyplot()
    with g2:
        # Display maximum heart rate achieved information
        st.write("Maximum Heart Rate Achieved")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('THALACH', bin=True),
            y='count()'
        ).properties(
            
        )
        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)



        
    g1, g2 = st.columns((4, 4))
    with g1:
        # Display age information
        st.write("Patient Age")
        # Create histogram
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('AGE', bin=alt.Bin(step=5)),
            y='count()',
        )
        # Display chart in Streamlit
        st.altair_chart(chart, use_container_width=True)
                
    with g2:
        # Display chest pain type information
        st.write("Chest Pain Type")
        cp_count = df['CP'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(cp_count, labels=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
        
    g1, g2 = st.columns((4, 4))
    with g1:
        st.write("Resting Blood Pressure")
        fig, ax = plt.subplots()
        ax.hist(df['TRESTBPS'], bins=10)
        ax.set_xlabel('Trestbps')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Trestbps')
        st.pyplot(fig)
    with g2:
        # Display fasting blood sugar information
        st.write("Fasting Blood Sugar")
        fbs_count = df['FBS'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(fbs_count, labels=["<= 120 mg/dl", "> 120 mg/dl"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)

    g1, g2, g3 = st.columns((4, 4, 4))
    with g1:
        st.write("Resting Electrocardiographic Results")
        restecg_count = df['RESTECG'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(restecg_count, labels=["Normal", "ST-wave-abnormality", "Probable"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)        
    with g2:
        # Display exercise induced angina information
        st.write("Exercise Induced Angina")
        exang_count = df['EXANG'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(exang_count, labels=["No", "Yes"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
    with g3:
        # Display the slope of the peak exercise ST segment information
        st.write("Slope of Peak Exercise ST Segment")
        slope_count = df['SLOPE'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(slope_count, labels=["Upsloping", "Flat", "Downsloping"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
    
    query = "SELECT * FROM FEATURE_IMPORTANCE;"
    cur = ctx.cursor().execute(query)
    FEATURE_IMPORTANCE_df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
    # sort the DataFrame by "IMPORTANCE" in descending order
    df_sorted = FEATURE_IMPORTANCE_df.sort_values('IMPORTANCE', ascending=False)
    st.set_option('deprecation.showPyplotGlobalUse', False) #disable pyplot old version warning msg
    #st.subheader('Feature Importance')
    st.write('The following features were found to be the most important in predicting heart health status:')
    # set the size of the bar graph
    fig, ax = plt.subplots(figsize=(15, 5))
    # create the bar graph using seaborn
    sns.barplot(x='IMPORTANCE', y='FEATURE', data=df_sorted, ax=ax)
    # set the title and axis labels
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    # display the bar graph in the Streamlit app
    st.pyplot(fig)

# Define the function to display the prediction page
def prediction():
    with st.form(key='prediction_form'):
        g1, g2, g3, g4, g5, g6, g7 = st.columns((2, 2, 2, 2, 2, 2, 2))
        with g1:
            AGE = st.text_input('AGE')
            SEX = st.selectbox('GENDER',('Male', 'Female'))
            if SEX == 'Male':
                SEX = '1'
            else:
                SEX = '0'
        with g2:
            #CP = st.text_input('CP')
            CP = st.selectbox('CP',('Typical angina', 'Non anginal','Atypical angina', 'Asymptomatic'))
            if CP == 'Typical angina':
                CP = '0'
            elif CP == 'Non anginal':
                CP = '1'
            elif CP == 'Atypical angina':
                CP = '2'
            else:
                CP = '3'
            #typical angina 0, non anginal 1, Atypical angina 2, Asymptomatic 3
            TRESTBPS = st.text_input('TRESTBPS')
        with g3:
            CHOL = st.text_input('CHOL')
            #FBS = st.text_input('FBS')
            FBS = st.selectbox('FBS',('True', 'False'))
            if FBS == 'True':
                FBS = '1'
            else:
                FBS = '0'
        with g4:
            #RESTECG = st.text_input('RESTECG')
            RESTECG = st.selectbox('RESTECG',('ST-wave-abnormality', 'Normal','Probale'))
            if RESTECG == 'ST-wave-abnormality':
                RESTECG = '0'
            elif RESTECG == 'Normal':
                RESTECG = '1'
            else:
                RESTECG = '2'
            #ST-wave-abnormality 0, Normal 1, Probale 2
            THALACH = st.text_input('THALACH')
        with g5:
            #EXANG = st.text_input('EXANG')
            EXANG = st.selectbox('EXANG',('Yes', 'No'))
            if EXANG == 'Yes':
                EXANG = '1'
            else:
                EXANG = '0'
            OLDPEAK = st.text_input('OLDPEAK')
        with g6:
            SLOPE = st.text_input('SLOPE')
            CA = st.text_input('CA')
        with g7:
            THAL = st.text_input('THAL')
            #st.markdown('<br><br>        <a href="sx.s" target="_blank"> &nbsp;&nbsp;&nbsp;&nbsp;Know More&nbsp;&nbsp;&nbsp;&nbsp;</a>', unsafe_allow_html=True)

        # apply CSS styling to the submit button to adjust its width, alignment and color
        st.markdown(
            f"""
            <style>
                div.stButton > button:first-child {{
                    width: 127px;
                    background-color: Black;
                    margin: 0 auto;
                    display: block;
                    text-align: center;
                    justify-content: center;
                    align-items: center;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    border: none;
                    height: 40px;
                    font-size: 99px;
                    cursor: pointer;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
        if st.form_submit_button('Predict'):
            if not AGE or not SEX or not CP or not TRESTBPS or not CHOL or not FBS or not RESTECG or not THALACH or not EXANG or not OLDPEAK or not THAL:
                st.error('Please fill in all the fields.')
            else:        
                query = f'INSERT INTO INPUT_DATA VALUES ({AGE}, {SEX}, {CP}, {TRESTBPS}, {CHOL}, {FBS}, {RESTECG}, {THALACH}, {EXANG}, {OLDPEAK}, {SLOPE}, {CA}, {THAL})'
                ctx.cursor().execute(query)
                ctx.cursor().execute('call train_and_predict()')
                query = f"SELECT  * from FINAL_DATA ORDER BY TIME_STAMP DESC LIMIT 1"
                cur = ctx.cursor().execute(query)
                pred_df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
                Prediction_df = pred_df.loc[:, ['LOGISTICREGRESSION', 'GAUSSIANNB', 'KNEIGHBORSCLASSIFIER', 'DECISIONTREECLASSIFIER', 'RANDOMFORESTCLASSIFIER']]  # select columns by label
                Accuracy_df = pred_df.loc[:, ['LOGISTICREGRESSION_ACCURACY', 'GAUSSIANNB_ACCURACY', 'KNEIGHBORSCLASSIFIER_ACCURACY', 'DECISIONTREECLASSIFIER_ACCURACY', 'RANDOMFORESTCLASSIFIER_ACCURACY']]  # select columns by label
                #st.write("Prediction given by Models with Accuracy")
                # replace values
                Prediction_df = Prediction_df.replace({'0': 'Unhealty', '1': 'Healthy'})

                # horizontally concatenate the dataframes
                # transpose the dataframe
                Prediction_df = Prediction_df.transpose().reset_index().rename(columns={'index': 'Model_Name', 0: 'Prediction'})
                Accuracy_df = Accuracy_df.transpose().reset_index().rename(columns={'index': 'Model_Name', 0: 'Accuracy'})
                Accuracy_df = Accuracy_df['Accuracy']
                result_df = pd.concat([Prediction_df, Accuracy_df], axis=1)
                result_df_sorted = result_df.sort_values('Accuracy', ascending=False)
                #st.write(result_df)
                


                # Create the chart
                chart = alt.Chart(result_df).mark_bar().encode(
                    x=alt.X('Accuracy:Q', axis=alt.Axis(title='Accuracy')),
                    y=alt.Y('Model_Name:N', sort='-x', axis=alt.Axis(title='Model Name')),
                    color=alt.Color('Prediction:N', scale=alt.Scale(domain=['Healthy', 'Unhealty'], range=['#2ca02c', '#d62728'])),
                    tooltip=['Model_Name', 'Prediction', 'Accuracy']
                ).properties(
                    title='Model Prediction and Accuracy'
                )

                # Show the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

                

            
# Define the function to display the reference page
def reference():
    g1, g2, g3    = st.columns((2, 10, 2))
    with g2:
        st.markdown("<h4 style='text-align: center; color:#0020C2;'>References</h4>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("↪ Reference Research paper: ")
        st.write("[https://www.ijrte.org/wp-content/uploads/papers/v8i2S3/B11630782S319.pdf](https://www.ijrte.org/wp-content/uploads/papers/v8i2S3/B11630782S319.pdf)")
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("↪ Kaggle Dataset used:")
        st.write(" [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)")
        st.markdown("<hr>", unsafe_allow_html=True)

        #url=''
        #st.write(f'<iframe src="{url}" width="700" height="1000"></iframe>', unsafe_allow_html=True)
        

def architecture():
    st.markdown("<h5 style='text-align: center; color:#0020C2;'>Technologies used</h5>", unsafe_allow_html=True)
    g1, g2, g3    = st.columns((4, 10, 5))
    with g2:
        st.image('https://raw.githubusercontent.com/sagar-w/Voice_and_Gesture_Controlled-Robotic-Car/main/tech2.PNG', width=550)
    st.write("↪ SNOWFLAKE: ")
    st.write("Snowflake for efficient data storage")
    st.write("↪ SNOWPARK: ")
    st.write("Utilized Snowpark for training our machine learning models")
    st.write("↪ STREAMLIT: ")
    st.write("Used Streamlit for deploying, visualizing these models in a user-friendly manner and insightful data analysis")
    g1, g2, g3    = st.columns((2, 10, 5))
    with g2:
        st.image('https://raw.githubusercontent.com/sagar-w/Voice_and_Gesture_Controlled-Robotic-Car/main/Arch.PNG', width=650)

    st.markdown("<h6 style='text-align: left; color: #000066;'>Please explore the various features and capabilities of our application and let us know if you have any feedback or recommendations for enhancements.</h6>", unsafe_allow_html=True)
    

# Define the pages dictionary to map page names to functions
pages = {
    'Home': home,
    'Prediction': prediction,
    'Architecture':architecture,
    'References': reference
}



# Define the sidebar with links to the different pages
st.sidebar.title('ML model on Snowflake🎡')
page = st.sidebar.selectbox('Select a page', list(pages.keys()))

# Add an image to the sidebar
st.sidebar.image('https://globetechcdn.com/hospimedica/images/stories/articles/article_images/2022-07-15/SDD-294793891.jpg', width=297)

# Get the appropriate function from the pages dictionary based on the selected page
selected_page = pages[page]

# Call the selected function to display the page
selected_page()
