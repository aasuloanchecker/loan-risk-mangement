import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# قراءة البيانات
data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# تجهيز البيانات: نستخدم الراتب، مبلغ القرض، مدة القرض
X = data[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].fillna(0)
y = data['Loan_Status'].map({'Y': 1, 'N': 0})

# تدريب نموذج Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# تصميم واجهة Streamlit
st.title('Loan Risk Prediction App')

st.write('أدخل المعلومات التالية لمعرفة مدى خطورة القرض')

# مدخلات المستخدم
age = st.number_input('العمر', min_value=18, max_value=100)
income = st.number_input('الراتب الشهري (Applicant Income)', min_value=0)
loan_amount = st.number_input('مبلغ القرض المطلوب (Loan Amount)', min_value=0)
loan_term_years = st.number_input('مدة القرض (بالسنوات)', min_value=1, max_value=30)
loan_purpose = st.selectbox('سبب القرض', ['شراء منزل', 'شراء سيارة', 'دراسة', 'توسعة عمل', 'أخرى'])

# نحول مدة القرض من سنة إلى شهر عشان تناسب الداتا الأصلية
loan_term_months = loan_term_years * 12

# زر التوقع
if st.button('توقع نتيجة القرض'):
    prediction = model.predict([[income, loan_amount, loan_term_months]])
    
    if prediction[0] == 1:
        st.success('القرض: منخفض الخطورة (تمت الموافقة عليه)')
    else:
        st.error('القرض: عالي الخطورة (تم رفضه)')
