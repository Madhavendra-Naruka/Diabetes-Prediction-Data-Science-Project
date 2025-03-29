import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

scaler=MinMaxScaler()

# Load the trained model
loaded_model = joblib.load('diabetes_decision_tree_model.pkl')

# Define the Streamlit app
def main():
    st.title('Diabetes Prediction App')

    # User input for HbA1c and blood glucose levels
    HbA1c = st.number_input('Enter HbA1c:', min_value=0.0, max_value=20.0, value=7.0)
    blood_glucose = st.number_input('Enter blood glucose:', min_value=0.0, max_value=500.0, value=120.0)

    # Predict button
    if st.button('Predict'):
        HbA1c=min_max_normalize(HbA1c,3.5,9)
        blood_glucose=min_max_normalize(blood_glucose,80,300)
        print(HbA1c)
        print(blood_glucose)
        # Create a new data point as a 2D array (expected by the model)
        new_data_point = [[(HbA1c + blood_glucose) * (1/0.82)]]

        # Predict using the loaded model
        prediction = loaded_model.predict(new_data_point)

        # Display prediction result
        if prediction[0] == 1:
            st.write("The patient is predicted to have diabetes.")
        else:
            st.write("The patient is predicted not to have diabetes.")

if __name__ == '__main__':
    main()
