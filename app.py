import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Load the pre-trained model
breast_cancer_model = joblib.load('models/breast_cancer.sav')

# Load the pre-trained model
chronic_disease_model = joblib.load('models/chronic_model.sav')

# Load the hepatitis prediction model
hepatitis_model = joblib.load('models/hepititisc_model.sav')


liver_model = joblib.load('models/liver_model.sav')# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Add explanatory text with improved styling - Moving this function to the top
def add_chart_explanation(chart_type, metrics):
    st.markdown("""<div style='background-color: #f0f2f6; padding: 1em; border-radius: 5px; margin-top: 0.5em;'>""", unsafe_allow_html=True)
    
    if chart_type == "gauge":
        st.markdown("""
        <h4 style='font-size: 18px; color: #1f77b4;'>Understanding the Risk Score:</h4>
        <ul style='font-size: 16px; line-height: 1.5;'>
            <li><span style='color: green; font-weight: bold;'>0-30: Low Risk</span></li>
            <li><span style='color: #ffd700; font-weight: bold;'>30-70: Medium Risk</span></li>
            <li><span style='color: salmon; font-weight: bold;'>70-100: High Risk</span></li>
        </ul>
        """, unsafe_allow_html=True)
    elif chart_type == "bar":
        st.markdown(f"""
        <h4 style='font-size: 18px; color: #1f77b4;'>Understanding the Metrics:</h4>
        <ul style='font-size: 16px; line-height: 1.5;'>
            <li><span style='color: green; font-weight: bold;'>Green bars</span> indicate values within normal range</li>
            <li>Values outside the normal range may require attention</li>
            <li>Metrics shown: <span style='font-weight: bold;'>{', '.join(metrics)}</span></li>
        </ul>
        """, unsafe_allow_html=True)
    elif chart_type == "pie":
        st.markdown("""
        <h4 style='font-size: 18px; color: #1f77b4;'>Understanding the Distribution:</h4>
        <ul style='font-size: 16px; line-height: 1.5;'>
            <li>Size of each slice shows relative importance</li>
            <li>Click legend items to focus on specific factors</li>
            <li>Hover over slices for detailed information</li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to create gauge chart with adjusted text sizes
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 28, 'color': '#1f77b4', 'family': 'Arial, sans-serif'}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}, 'font': {'size': 16}},
        gauge = {
            'axis': {
                'range': [0, 100], 
                'tickwidth': 1, 
                'tickcolor': "darkblue",
                'ticktext': ['Low Risk', 'Medium Risk', 'High Risk'],
                'tickvals': [20, 50, 80],
                'tickfont': {'size': 14}
            },
            'bar': {'color': "darkblue", 'thickness': 0.6},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "lightgreen", 'name': 'Low Risk'},
                {'range': [30, 70], 'color': "yellow", 'name': 'Medium Risk'},
                {'range': [70, 100], 'color': "salmon", 'name': 'High Risk'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    fig.add_annotation(
        text=f"Risk Level: {'High' if value >= 70 else 'Medium' if value >= 30 else 'Low'}",
        x=0.5,
        y=0.25,
        showarrow=False,
        font=dict(size=20, color='darkblue', family='Arial, sans-serif')
    )
    fig.update_layout(
        paper_bgcolor = "white",
        height=450,  # Increased height
        margin=dict(l=20, r=20, t=100, b=20),  # Increased top margin for title
        font={'color': "darkblue", 'family': "Arial, sans-serif", 'size': 16}
    )
    return fig

# Function to create pie chart with improved readability and text sizes
def create_pie_chart(values, labels, title):
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
              
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.4,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=16, color='black', family='Arial, sans-serif'),
        pull=[0.1]*len(values),
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>" +
                      "Value: %{value}<br>" +
                      "Percentage: %{percent}<br>" +
                      "<extra></extra>",
        direction='clockwise',
        sort=False
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=28, color='#1f77b4', family='Arial, sans-serif')
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14, family='Arial, sans-serif')
        ),
        height=500,
        margin=dict(l=50, r=50, t=100, b=100),  # Increased margins
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=[
            dict(
                text="Click legend items to filter",
                x=0.5,
                y=-0.3,
                showarrow=False,
                font=dict(size=14, family='Arial, sans-serif'),
                xref="paper",
                yref="paper"
            )
        ]
    )
    return fig

# Function to create bar chart with improved text sizes
def create_bar_chart(x_values, y_values, title, normal_ranges=None):
    fig = go.Figure()
    
    # Add the main bar chart
    fig.add_trace(go.Bar(
        x=x_values, 
        y=y_values,
        text=y_values,
        textposition='auto',
        textfont=dict(size=14, family='Arial, sans-serif'),
        marker_color='darkblue',
        opacity=0.8,
        name='Current Values',
        hovertemplate="<b>%{x}</b><br>" +
                      "Value: %{y:.2f}<br>" +
                      "<extra></extra>"
    ))
    
    # Add normal range indicators if provided
    if normal_ranges:
        for x, (min_val, max_val) in zip(x_values, normal_ranges):
            fig.add_trace(go.Scatter(
                x=[x, x],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='green', width=3),
                name=f'Normal Range ({min_val}-{max_val})',
                showlegend=False
            ))
            fig.add_annotation(
                x=x,
                y=max_val,
                text=f'Normal<br>Range',
                showarrow=False,
                yshift=10,
                font=dict(size=12, family='Arial, sans-serif')
            )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=28, color='#1f77b4', family='Arial, sans-serif')
        ),
        xaxis=dict(
            title='',
            tickangle=45,
            tickfont=dict(size=14, family='Arial, sans-serif'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text='Value', font=dict(size=16, family='Arial, sans-serif')),
            tickfont=dict(size=14, family='Arial, sans-serif'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        height=450,  # Increased height
        margin=dict(l=20, r=20, t=100, b=100),  # Increased margins
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14, family='Arial, sans-serif')
        )
    )
    return fig

# Function to add spacing between visualizations
def add_vertical_space():
    st.markdown("<div style='margin: 1em 0em;'></div>", unsafe_allow_html=True)

# Function to create container with proper spacing
def create_visualization_container():
    return st.container()

# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction',
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Lung Cancer Prediction',
        'Chronic Kidney prediction',
        'Breast Cancer Prediction',

    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)




# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')

        with create_visualization_container():
            # First row of visualizations
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(create_gauge_chart(prob*100, "Prediction Probability"), use_container_width=True)
                add_chart_explanation("gauge", [])
            with col2:
                st.plotly_chart(create_pie_chart([1]*len(symptoms), symptoms, "Symptoms Distribution"), use_container_width=True)
                add_chart_explanation("pie", [])
            
            add_vertical_space()

        tab1, tab2, tab3 = st.tabs(["Description", "Precautions", "Visualization"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')
                
        with tab3:
            with create_visualization_container():
                st.plotly_chart(create_pie_chart([1]*len(symptoms), symptoms, "Symptoms Distribution"), use_container_width=True)




# Diabetes prediction page
if selected == 'Diabetes Prediction':  # pagetitle
    st.title("Diabetes disease prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='diabetes disease prediction')
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure  value")
    with col1:

        SkinThickness = st.number_input("Sckinthickness value")

    with col2:

        Insulin = st.number_input("Insulin value ")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input(
            "Diabetespedigreefunction value")
    with col2:

        Age = st.number_input("AGE")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("Diabetes test result"):
        # Initialize variables
        risk_score = 0
        metrics = ['Glucose', 'BMI', 'Blood Pressure']
        values = [Glucose, BMI, BloodPressure]
        normal_ranges = [(70, 100), (18.5, 24.9), (90, 120)]  # Normal ranges for each metric
        
        # Make prediction
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])
        
        # Update risk score after prediction
        risk_score = diabetes_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Diabetes Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Health Metrics", normal_ranges), use_container_width=True)
            
            add_vertical_space()

        # Display prediction result
        if diabetes_prediction[0] == 1:
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + diabetes_dig)
        
        



# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='heart failuire')
    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            sex = 1
        elif value == "female":
            sex = 0
    with col3:
        cp=0
        display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
        if value == "typical angina":
            cp = 0
        elif value == "atypical angina":
            cp = 1
        elif value == "non — anginal pain":
            cp = 2
        elif value == "asymptotic":
            cp = 3
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:

        chol = st.number_input("Serum Cholestrol")
    
    with col3:
        restecg=0
        display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if value == "normal":
            restecg = 0
        elif value == "having ST-T wave abnormality":
            restecg = 1
        elif value == "left ventricular hyperthrophy":
            restecg = 2

    with col1:
        exang=0
        thalach = st.number_input("Max Heart Rate Achieved")
   
    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col3:
        slope=0
        display = ("upsloping","flat","downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
        if value == "upsloping":
            slope = 0
        elif value == "flat":
            slope = 1
        elif value == "downsloping":
            slope = 2
    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by flourosopy")
    with col2:
        thal=0
        display = ("normal","fixed defect","reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
        if value == "normal":
            thal = 0
        elif value == "fixed defect":
            thal = 1
        elif value == "reversible defect":
            thal = 2
    with col3:
        agree = st.checkbox('Exercise induced angina')
        if agree:
            exang = 1
        else:
            exang=0
    with col1:
        agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
        if agree1:
            fbs = 1
        else:
            fbs=0
    # code for prediction
    heart_dig = ''
    

    # button
    if st.button("Heart test result"):
        # Initialize variables
        risk_score = 0
        metrics = ['Blood Pressure', 'Cholesterol', 'Max Heart Rate']
        values = [trestbps, chol, thalach]
        normal_ranges = [(90, 120), (125, 200), (60, 100)]  # Normal ranges for each metric
        
        # Make prediction
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Update risk score after prediction
        risk_score = heart_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Heart Disease Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Cardiac Metrics", normal_ranges), use_container_width=True)
            
            add_vertical_space()

        if heart_prediction[0] == 1:
            heart_dig = 'we are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            
        else:
            heart_dig = "Congratulation , You don't have Heart Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name +' , ' + heart_dig)









if selected == 'Parkison Prediction':
    st.title("Parkison prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')
  # parameters
#    name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
   # change the variables according to the dataset used in the model
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.number_input("MDVP:Fo(Hz)")
    with col2:
        MDVPFIZ = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        MDVPFLO = st.number_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJITTER = st.number_input("MDVP:Jitter(%)")
    with col2:
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        MDVPRAP = st.number_input("MDVP:RAP")

    with col2:

        MDVPPPQ = st.number_input("MDVP:PPQ ")
    with col3:
        JitterDDP = st.number_input("Jitter:DDP")
    with col1:
        MDVPShimmer = st.number_input("MDVP:Shimmer")
    with col2:
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
    with col1:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5")
    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ")
    with col3:
        ShimmerDDA = st.number_input("Shimmer:DDA")
    with col1:
        NHR = st.number_input("NHR")
    with col2:
        HNR = st.number_input("HNR")
  
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col1:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    # code for prediction
    parkinson_dig = ''
    
    # button
    if st.button("Parkinson test result"):
        # Initialize variables
        risk_score = 0
        metrics = ['Jitter', 'Shimmer', 'NHR', 'HNR']
        values = [MDVPJITTER, MDVPShimmer, NHR, HNR]
        
        # Make prediction
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        
        # Update risk score after prediction
        risk_score = parkinson_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Parkinson's Disease Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Voice Metrics"), use_container_width=True)
            
            add_vertical_space()

        # Display prediction result
        if parkinson_prediction[0] == 1:
            parkinson_dig = 'we are really sorry to say but it seems like you have Parkinson disease'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            parkinson_dig = "Congratulation , You don't have Parkinson disease"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + parkinson_dig)



# Load the dataset
lung_cancer_data = pd.read_csv('data/lung_cancer.csv')

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Lung Cancer Prediction')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    cancer_result = ''

    # Button
    if st.button("Predict Lung Cancer"):
        # Initialize variables
        risk_score = 0
        risk_factors = ['Smoking', 'Alcohol', 'Chronic Disease', 'Chest Pain']
        values = [2 if smoking == 'YES' else 1, 
                 2 if alcohol_consuming == 'YES' else 1,
                 2 if chronic_disease == 'YES' else 1,
                 2 if chest_pain == 'YES' else 1]
        
        # Create DataFrame and make prediction
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # Map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)
        user_data.columns = user_data.columns.str.strip()
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        cancer_prediction = lung_cancer_model.predict(user_data)
        
        # Update risk score after prediction
        risk_score = 100 if cancer_prediction[0] == 'YES' else 0

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Lung Cancer Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_pie_chart(values, risk_factors, "Risk Factors Distribution"), use_container_width=True)
            
            add_vertical_space()

        # Display prediction result
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + cancer_result)




# Liver prediction page
if selected == 'Liver prediction':  # pagetitle
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 
    # code for prediction
    liver_dig = ''

    # button
    if st.button("Liver test result"):
        # Initialize variables
        risk_score = 0
        metrics = ['Total Bilirubin', 'Direct Bilirubin', 'Albumin']
        values = [Total_Bilirubin, Direct_Bilirubin, Albumin]
        normal_ranges = [(0.3, 1.2), (0.1, 0.3), (3.5, 5.5)]  # Normal ranges for each metric
        
        # Make prediction
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
        
        # Update risk score after prediction
        risk_score = liver_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Liver Disease Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Liver Function Tests", normal_ranges), use_container_width=True)
            
            add_vertical_space()

        # Display prediction result
        if liver_prediction[0] == 1:
            liver_dig = "we are really sorry to say but it seems like you have liver disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            liver_dig = "Congratulation , You don't have liver disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + liver_dig)






# Hepatitis prediction page
if selected == 'Hepatitis prediction':
    st.title("Hepatitis Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Hepatitis Prediction')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter your age")  # 2
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 2
    with col3:
        total_bilirubin = st.number_input("Enter your Total Bilirubin")  # 3

    with col1:
        direct_bilirubin = st.number_input("Enter your Direct Bilirubin")  # 4
    with col2:
        alkaline_phosphatase = st.number_input("Enter your Alkaline Phosphatase")  # 5
    with col3:
        alamine_aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  # 6

    with col1:
        aspartate_aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  # 7
    with col2:
        total_proteins = st.number_input("Enter your Total Proteins")  # 8
    with col3:
        albumin = st.number_input("Enter your Albumin")  # 9

    with col1:
        albumin_and_globulin_ratio = st.number_input("Enter your Albumin and Globulin Ratio")  # 10

    with col2:
        your_ggt_value = st.number_input("Enter your GGT value")  # Add this line
    with col3:
        your_prot_value = st.number_input("Enter your PROT value")  # Add this line

    # Code for prediction
    hepatitis_result = ''

    # Button
    if st.button("Predict Hepatitis"):
        # Initialize variables
        risk_score = 0
        metrics = ['Total Bilirubin', 'Direct Bilirubin', 'Albumin']
        values = [total_bilirubin, direct_bilirubin, albumin]
        
        # Create DataFrame for prediction
        user_input_hepatitis = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ALB': [total_bilirubin],
            'ALP': [direct_bilirubin],
            'ALT': [alkaline_phosphatase],
            'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase],
            'CHE': [total_proteins],
            'CHOL': [albumin],
            'CREA': [albumin_and_globulin_ratio],
            'GGT': [your_ggt_value],
            'PROT': [your_prot_value]
        })

        # Make prediction
        hepatitis_prediction = hepatitis_model.predict(user_input_hepatitis)
        
        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                risk_score = 100 if hepatitis_prediction[0] == 1 else 0
                st.plotly_chart(create_gauge_chart(risk_score, "Hepatitis Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Liver Function Tests"), use_container_width=True)
            
            add_vertical_space()

        # Display prediction result
        if hepatitis_prediction[0] == 1:
            hepatitis_result = "We are really sorry to say but it seems like you have Hepatitis."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            hepatitis_result = 'Congratulations, you do not have Hepatitis.'
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + hepatitis_result)











# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney prediction':
    st.title("Chronic Kidney Disease Prediction")
    # Add the image for Chronic Kidney Disease prediction if needed
    name = st.text_input("Name:")
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Enter your age", 1, 100, 25)  # 2
    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120)  # Add your own ranges
    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)  # Add your own ranges

    with col1:
        al = st.slider("Enter your Albumin", 0, 5, 0)  # Add your own ranges
    with col2:
        su = st.slider("Enter your Sugar", 0, 5, 0)  # Add your own ranges
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc = 1 if rbc == "Normal" else 0

    with col1:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        pc = 1 if pc == "Normal" else 0
    with col2:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        pcc = 1 if pcc == "Present" else 0
    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        ba = 1 if ba == "Present" else 0

    with col1:
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)  # Add your own ranges
    with col2:
        bu = st.slider("Enter your Blood Urea", 10, 200, 60)  # Add your own ranges
    with col3:
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)  # Add your own ranges

    with col1:
        sod = st.slider("Enter your Sodium", 100, 200, 140)  # Add your own ranges
    with col2:
        pot = st.slider("Enter your Potassium", 2, 7, 4)  # Add your own ranges
    with col3:
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)  # Add your own ranges

    with col1:
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)  # Add your own ranges
    with col2:
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)  # Add your own ranges
    with col3:
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)  # Add your own ranges

    with col1:
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        htn = 1 if htn == "Yes" else 0
    with col2:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        dm = 1 if dm == "Yes" else 0
    with col3:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        cad = 1 if cad == "Yes" else 0

    with col1:
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet = 1 if appet == "Good" else 0
    with col2:
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        pe = 1 if pe == "Yes" else 0
    with col3:
        ane = st.selectbox("Anemia", ["Yes", "No"])
        ane = 1 if ane == "Yes" else 0

    # Code for prediction
    kidney_result = ''

    # Button
    if st.button("Predict Chronic Kidney Disease"):
        # Initialize variables
        risk_score = 0
        metrics = ['Blood Pressure', 'Blood Glucose', 'Hemoglobin']
        values = [bp, bgr, hemo]
        
        # Create DataFrame for prediction
        user_input_kidney = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })
        
        # Make prediction
        kidney_prediction = chronic_disease_model.predict(user_input_kidney)
        
        # Update risk score after prediction
        risk_score = kidney_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Chronic Kidney Disease Risk Score"), use_container_width=True)
                add_chart_explanation("gauge", [])
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Kidney Function Metrics"), use_container_width=True)
                add_chart_explanation("bar", metrics)
            
            add_vertical_space()

        # Additional visualization for blood composition
        blood_metrics = ['White Blood Cells', 'Red Blood Cells', 'Packed Cell Volume']
        blood_values = [wc/1000, rc, pcv] # Normalize WBC count
        st.plotly_chart(create_bar_chart(blood_metrics, blood_values, "Blood Composition Metrics"), use_container_width=True)
        add_chart_explanation("bar", blood_metrics)

        # Display prediction result
        if kidney_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "we are really sorry to say but it seems like you have kidney disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "Congratulation , You don't have kidney disease."
        st.success(name+' , ' + kidney_prediction_dig)



# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")
    name = st.text_input("Name:")
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

        # Code for prediction
    breast_cancer_result = ''

    # Button
    if st.button("Predict Breast Cancer"):
        # Initialize variables
        risk_score = 0
        metrics = ['Radius', 'Texture', 'Perimeter', 'Area']
        values = [radius_mean, texture_mean, perimeter_mean, area_mean]
        
        # Create DataFrame for prediction
        user_input_breast = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst]
        })
        
        # Make prediction
        breast_cancer_prediction = breast_cancer_model.predict(user_input_breast)
        
        # Update risk score after prediction
        risk_score = breast_cancer_prediction[0] * 100

        # Create visualizations
        with create_visualization_container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_score, "Breast Cancer Risk Score"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_bar_chart(metrics, values, "Key Tumor Characteristics"), use_container_width=True)
            
            add_vertical_space()

        # Additional visualization for cell characteristics
        cell_metrics = ['Smoothness', 'Compactness', 'Concavity', 'Symmetry']
        cell_values = [smoothness_mean, compactness_mean, concavity_mean, symmetry_mean]
        st.plotly_chart(create_bar_chart(cell_metrics, cell_values, "Cell Characteristics"), use_container_width=True)

        # Display prediction result
        if breast_cancer_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."

        st.success(breast_cancer_result)
