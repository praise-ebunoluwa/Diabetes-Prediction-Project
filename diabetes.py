import streamlit as st
import numpy as np

import pickle

st.set_page_config(page_title="Diabetes Prediction App")

def age():
    with open("label_age.pkl", "rb") as f:
        new_age = pickle.load(f)
    return new_age

def bmi():
    with open("label_bmi.pkl", "rb") as f:
        new_bmi = pickle.load(f)
    return new_bmi

def glucose():
    with open("label_glucose.pkl", "rb") as f:
        new_glucose = pickle.load(f)
    return new_glucose

def outcome():
    with open("label_outcome.pkl", "rb") as f:
        new_outcome = pickle.load(f)
    return new_outcome

def bp():
    with open("label_bp.pkl", "rb") as f:
        new_blood_pressure = pickle.load(f)
    return new_blood_pressure

def model():
    with open("rfc.pkl", "rb") as f:
        new_rfc = pickle.load(f)
    return new_rfc


def scaler():
    with open("scaler.pkl", "rb") as f:
        new_scale = pickle.load(f)
    return new_scale

new_label_age = age()
new_label_bmi = bmi()
new_label_glucose = glucose()
new_label_outcome = outcome()
new_label_bp = bp()
random_forest = model()
scale = scaler()

def app():

    st.sidebar.title("Navigation")
    buton = st.sidebar.radio("Go to ", ["Home", "Result"])
    if buton == "Home":
        st.title("DIABETES RISK PREDICTION APP FOR FEMALES")
        # st.header("Welcome")
        st.subheader("Predict your likelihood of having diabetes based on health data")
        st.markdown("""
        This app uses a trained machine learning model to predict whether a female is likely to have diabetes, based on personal health indicators like age, pregnancies, BMI, family history, insulin level,and glucose level.

        You can use it to:
        - Understand your risk level
        - Explore how different health factors affect predictions
        - Learn more about diabetes risk factors
        """)
        st.markdown(""" 
                    Major Diabetes Risk Factors
                    
        - Genetics / Family History: Having a close relative (parent or sibling) with diabetes increases your risk. Often captured through Diabetes Pedigree Function in models.
        - Age: Risk increases with age, especially after age 45.
        - High Blood Pressure (Hypertension): Often associated with insulin resistance and metabolic syndrome.
        - High Blood Sugar (Glucose): Pre-diabetic glucose levels (100–125 mg/dL fasting) are a major warning sign.
        """)
    elif buton == "Result":

        st.header("Fill the information and get your result")

        Pregnancies = st.number_input("Please enter the number of times pregnant")
        Age = st.number_input("please enter your age")
        Glucose = st.number_input("please enter your glucose")
        BloodPressure = st.number_input("please enter your bloodpressure")
        SkinThickness = st.number_input("please enter your skinthickness ")
        Insulin = st.number_input("please enter the value of your insulin")
        BMI = st.number_input("please enter your body mass index")
        NUMBER = st.number_input("what percentage of your family members have diabetes")
        if (NUMBER > 0):
            DiabetesPedigree = (NUMBER / 100)
        
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function (auto-filled from percentage)",value = DiabetesPedigree, min_value=0.0,max_value=1.0,step=0.01,format="%.2f")  
        

        if (BMI < 18.5):
            BMI =  "underweight"
        elif (BMI >= 18.5 and BMI <= 24.9):
            BMI = "healthyweight"
        elif (BMI >= 25 and BMI <= 29.9):
            BMI = "overweight"
        elif(BMI >= 30):
            BMI =  "obesity"
        
            
        if (Age >= 20 and Age <= 40):
            Age =  "youth"
        elif (Age>= 41 and Age<= 69):
            Age =  "adult"
        elif (Age >= 70):
            Age =  "old"

        if (BloodPressure <= 60):
            BloodPressure =  "low"
        elif (BloodPressure >= 61 and BloodPressure <= 79):
            BloodPressure =  "normal"
        elif (BloodPressure >= 80 and BloodPressure <= 89):
            BloodPressure = "prohypertension"
        elif (BloodPressure >= 90 and BloodPressure <= 99):
            BloodPressure =  "stage1"
        elif (BloodPressure >= 100 and BloodPressure <=  109):
            BloodPressure =  "stage2"
        elif (BloodPressure >= 110):
            BloodPressure = "crisis"
        
        
        if (Glucose < 60):
            Glucose =  "hypoglycemia"
        elif (Glucose >= 60 and Glucose <= 70):
            Glucose = "early_hypoglycemia"
        elif (Glucose >= 71 and Glucose <= 140):
            Glucose = "normal"
        elif (Glucose >= 141 and Glucose <=200):
            Glucose = "early_diabetes"
        elif (Glucose >= 201):
            Glucose = "diabetic"

        Glucose_new = int(new_label_glucose.transform([Glucose]))
        BloodPressure_new = int(new_label_bp.transform([BloodPressure]))
        BMI_new = int(new_label_bmi.transform([BMI]))
        Age_new = int(new_label_age.transform([Age]))

        features = np.array([Pregnancies, Glucose_new, BloodPressure_new, SkinThickness, Insulin, BMI_new, DiabetesPedigreeFunction, Age_new])
        scalee = scale.transform(features.reshape(1, -1))
        result = random_forest.predict(scalee)

        if st.button("Predict"):

            if result == [0]:
                result = "diabetic"
            elif result == [1]:
                result =  "non diabetic"
            st.success("After thorough examination, it can be predicted that you are {}". format(result))


if __name__ == "__main__":
    app()
    



    
