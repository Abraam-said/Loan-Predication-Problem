
import joblib
import streamlit as st
import pandas as pd

Model = joblib.load("Model_Final.pkl")
Inputs = joblib.load("Inputs_Final.pkl")

def prediction(gender, married, dependents, education, self_employed,
               applicant_income, coapplicant_income, loan_amount,
               loan_amount_term, credit_history, property_area):
    # Create a test dataframe with the selected inputs
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0, 'Gender'] = gender
    test_df.at[0, 'Married'] = married
    test_df.at[0, 'Dependents'] = dependents
    test_df.at[0, 'Education'] = education
    test_df.at[0, 'Self_Employed'] = self_employed
    test_df.at[0, 'ApplicantIncome'] = applicant_income
    test_df.at[0, 'CoapplicantIncome'] = coapplicant_income
    test_df.at[0, 'LoanAmount'] = loan_amount
    test_df.at[0, 'Loan_Amount_Term'] = loan_amount_term
    test_df.at[0, 'Credit_History'] = credit_history
    test_df.at[0, 'Property_Area'] = property_area
    
    # Predict using the model and return the result
    result = Model.predict(test_df)
    return result[0]

def main():
    ## Setting up the page title and icon
    st.set_page_config(page_icon='💰', page_title='Loan Prediction App')

    # Add a title in the middle of the page using Markdown and CSS
    st.markdown("<h1 style='text-align: center;text-decoration: underline;color:White'>Loan Prediction App 💰</h1>",
                unsafe_allow_html=True)

    # Input fields for user interaction
    gender = st.radio('Gender', ['Male', 'Female'])
    
    married = st.radio("Married", ['No', 'Yes'])
    
    dependents = st.selectbox("Number of Dependents", ['0', '1', '2', '3+'])
    
    education = st.radio("Education", ['Graduate', 'Not Graduate'])
    
    self_employed = st.radio("Self Employed", ['No', 'Yes'])
    
    applicant_income = st.number_input("Applicant's Income", min_value=0, max_value=100000, value=0, step=100)
    
    coapplicant_income = st.number_input("Coapplicant's Income", min_value=0, max_value=100000, value=0, step=100)
    
    loan_amount = st.number_input("Loan Amount", min_value=10,  max_value=100000, value=10, step=5)
    
    loan_amount_term = st.selectbox("Loan Amount Term (in months)", [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360])
    
    credit_history_mapping = {'Good': 1.0, 'Not Good': 0.0}
    credit_history_label = st.radio("Credit History", list(credit_history_mapping.keys()))

    # Convert the selected label to the numerical encoding
    credit_history = credit_history_mapping[credit_history_label]
        
    property_area = st.selectbox("Property Area",
                                  options=['Urban', 'Rural', 'Semiurban'])
    # Predict button
    if st.button("Predict"):
        result = prediction(gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_amount_term, credit_history, property_area)
        if result == 1:
            st.success("Congratulations! The loan is likely to be approved.")
        else:
            st.error("Sorry, the loan is likely to be rejected.")

if __name__ == '__main__':
    main()
