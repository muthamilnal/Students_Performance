# import the libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

# load model
def load_model():
    with open("student_lr_model.pkl","rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le 

# preprocessing
def preprocessing_input_data(data,scaler,le):
    # label encoding on the user input for Extra curricular activities
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    # data is coming fromt he userinterface inthe form of dictionary
    # how to convert dict to dataframe
    df = pd.DataFrame([data])
    st.write(df)
    # now convert all the numbers in standard scalar format
    df_transformed = scaler.transform(df)
    st.write(df_transformed)

    # display both user input and transformed result : Dataframe
    result_data = {
        "Features":df.columns,
        "User given values":df.iloc[0].values,
        "Transformed values":df_transformed[0]
        
    }
    st.table(result_data)
    return df_transformed

# prediction
def predict_data(data):
    # data is nothing but it is from user interface
    # load the model 
    model,scaler,le = load_model()
    # preprocess the data
    processed_data = preprocessing_input_data(data,scaler,le)
    # make prediction 
    prediction = model.predict(processed_data)
    return prediction
# call predict_data() function inside streamlit app

def main():
    st.title("Student Performance Predition")
    st.write("Enter your data to predition")
    hours_studied = st.number_input("Hours studied",min_value=1,max_value=10,value=3)
    previous_score = st.number_input("Previous Score",min_value=40,max_value=100,value=50)
    extra = st.selectbox("Extra curricular Activities",["Yes","No"])
    sleeping_hours = st.number_input("Sleeping hours",min_value=4,max_value=10,value=5)
    no_papers_solved = st.number_input("No of papers solved",min_value=0,max_value=10,value=3)

    if st.button("PredictScore"):
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hours,
            "Sample Question Papers Practiced":no_papers_solved
        }
        # data is ready
        # we have to pass this data to the prediction function what we have defined
        result = predict_data(user_data)
        st.success(f"The Score : {result}")

if __name__ == "__main__":
    main()
