import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Read the dataset
data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Prepare the data: use Applicant Income, Loan Amount, Loan Term
X = data[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].fillna(0)
y = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit App
st.title('Loan Risk Prediction App')

st.write('Enter the following information to predict loan risk')

# User Inputs
age = st.number_input('Age', min_value=18, max_value=100)
income = st.number_input('Monthly Income (Applicant Income)', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_term_years = st.number_input('Loan Term (in years)', min_value=1, max_value=30)
loan_purpose = st.selectbox('Loan Purpose', ['Buy a House', 'Buy a Car', 'Education', 'Business Expansion', 'Other'])

# Convert loan term from years to months
loan_term_months = loan_term_years * 12

# Predict
if st.button('Predict Loan Result'):
    prediction = model.predict([[income, loan_amount, loan_term_months]])
    
    if prediction[0] == 1:
        st.success('Loan Approved (Low Risk)')
    else:
        st.error('Loan Rejected (High Risk)')