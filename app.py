import streamlit as st
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
import base64
import os

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

hide_st_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Update the background image path
background_image_path = os.path.join(os.path.dirname(__file__), 'assets', 'download (4).jpg')
try:
    background_encoded = get_base64_encoded_image(background_image_path)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{background_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.4);
        pointer-events: none;
    }}

    .stSelectbox, .stSlider, .stButton {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }}

    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0.5);
    }}

    .card {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }}

    h1, h2, h3, p, label, .stMarkdown {{
        color: white !important;
    }}

    .prediction-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        margin-top: 20px;
    }}

    .button-container {{
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
except Exception as e:
    st.warning("Please ensure the background image exists in the assets folder")

models = {
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb'))
}

selected = st.selectbox(
    'Select a Disease to Predict',
    ['Parkinsons Disease', 'Lung Cancer']
)

def display_input(label, tooltip, key, type="text"):
    if type == "text":
        return st.text_input(label, key=key, help=tooltip)
    elif type == "number":
        return st.number_input(label, key=key, help=tooltip, step=1)
    
if selected == 'Parkinsons Disease':
    st.title('Parkinsons Disease')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### MDVP Frequency Features")
        mdvp_fo = display_input('MDVP:Fo(Hz)', 'Average vocal fundamental frequency', 'mdvp_fo', 'number')
        mdvp_fhi = display_input('MDVP:Fhi(Hz)', 'Maximum vocal fundamental frequency', 'mdvp_fhi', 'number')
        mdvp_flo = display_input('MDVP:Flo(Hz)', 'Minimum vocal fundamental frequency', 'mdvp_flo', 'number')
        mdvp_jitter = display_input('MDVP:Jitter(%)', 'Frequency perturbation', 'mdvp_jitter', 'number')
        mdvp_jitter_abs = display_input('MDVP:Jitter(Abs)', 'Absolute frequency perturbation', 'mdvp_jitter_abs', 'number')
        mdvp_rap = display_input('MDVP:RAP', 'Relative amplitude perturbation', 'mdvp_rap', 'number')
        mdvp_ppq = display_input('MDVP:PPQ', 'Five-point period perturbation quotient', 'mdvp_ppq', 'number')
        jitter_ddp = display_input('Jitter:DDP', 'Average difference of differences', 'jitter_ddp', 'number')
    
    with col2:
        st.write("### MDVP Shimmer Features")
        mdvp_shimmer = display_input('MDVP:Shimmer', 'Amplitude perturbation', 'mdvp_shimmer', 'number')
        mdvp_shimmer_db = display_input('MDVP:Shimmer(dB)', 'Shimmer in dB', 'mdvp_shimmer_db', 'number')
        shimmer_apq3 = display_input('Shimmer:APQ3', 'Three-point amplitude perturbation', 'shimmer_apq3', 'number')
        shimmer_apq5 = display_input('Shimmer:APQ5', 'Five-point amplitude perturbation', 'shimmer_apq5', 'number')
        mdvp_apq = display_input('MDVP:APQ', 'Amplitude perturbation quotient', 'mdvp_apq', 'number')
        shimmer_dda = display_input('Shimmer:DDA', 'Average amplitude differences', 'shimmer_dda', 'number')
    
    with col3:
        st.write("### Additional Voice Measures")
        nhr = display_input('NHR', 'Noise-to-harmonics ratio', 'nhr', 'number')
        hnr = display_input('HNR', 'Harmonics-to-noise ratio', 'hnr', 'number')
        rpde = display_input('RPDE', 'Recurrence period density entropy', 'rpde', 'number')
        dfa = display_input('DFA', 'Detrended fluctuation analysis', 'dfa', 'number')
        spread1 = display_input('spread1', 'Nonlinear measure of fundamental frequency variation', 'spread1', 'number')
        spread2 = display_input('spread2', 'Nonlinear measure of fundamental frequency variation', 'spread2', 'number')
        d2 = display_input('D2', 'Correlation dimension', 'd2', 'number')
        ppe = display_input('PPE', 'Pitch period entropy', 'ppe', 'number')

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button('Predict Parkinson\'s Disease', type='primary'):
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        input_data = [[
            mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs,
            mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
            shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]]
        prediction = models['parkinsons'].predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that the patient may have Parkinson's Disease")
        else:
            st.success("✅ The model predicts that the patient is healthy")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == 'Lung Cancer':
    st.title('Lung Cancer')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### Personal Information")
        age = st.slider('Age', 20, 100, 50, help='Patient age')
        gender = st.selectbox('Gender', ['Male', 'Female'], key='gender')
        smoking = st.selectbox('Smoking', ['Yes', 'No'], key='smoking')
        alcohol = st.selectbox('Alcohol Consuming', ['Yes', 'No'], key='alcohol')
        peer_pressure = st.selectbox('Peer Pressure', ['Yes', 'No'], key='peer_pressure')
    
    with col2:
        st.write("### Primary Symptoms")
        chronic_disease = st.selectbox('Chronic Disease', ['Yes', 'No'], key='chronic_disease')
        fatigue = st.selectbox('Fatigue', ['Yes', 'No'], key='fatigue')
        allergy = st.selectbox('Allergy', ['Yes', 'No'], key='allergy')
        wheezing = st.selectbox('Wheezing', ['Yes', 'No'], key='wheezing')
        coughing = st.selectbox('Coughing', ['Yes', 'No'], key='coughing')
    
    with col3:
        st.write("### Secondary Symptoms")
        shortness_breath = st.selectbox('Shortness of Breath', ['Yes', 'No'], key='shortness_breath')
        swallowing = st.selectbox('Difficulty Swallowing', ['Yes', 'No'], key='swallowing')
        chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'], key='chest_pain')
        yellow_fingers = st.selectbox('Yellow Fingers', ['Yes', 'No'], key='yellow_fingers')
        anxiety = st.selectbox('Anxiety', ['Yes', 'No'], key='anxiety')

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button('Predict Lung Cancer', type='primary'):
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        input_data = [[
            1 if gender == 'Male' else 0,
            age,
            1 if smoking == 'Yes' else 0,
            1 if yellow_fingers == 'Yes' else 0,
            1 if anxiety == 'Yes' else 0,
            1 if peer_pressure == 'Yes' else 0,
            1 if chronic_disease == 'Yes' else 0,
            1 if fatigue == 'Yes' else 0,
            1 if allergy == 'Yes' else 0,
            1 if wheezing == 'Yes' else 0,
            1 if alcohol == 'Yes' else 0,
            1 if coughing == 'Yes' else 0,
            1 if shortness_breath == 'Yes' else 0,
            1 if swallowing == 'Yes' else 0,
            1 if chest_pain == 'Yes' else 0
        ]]
        prediction = models['lung_cancer'].predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that the patient may have Lung Cancer")
        else:
            st.success("✅ The model predicts that the patient is healthy")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.markdown("""
<div style='text-align: center'>
    <small>Disclaimer: This is a prototype application. Please consult with healthcare professionals for accurate medical diagnosis.</small>
</div>
""", unsafe_allow_html=True)
