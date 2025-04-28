import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# قراءة البيانات
data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# نختار الأعمدة المهمة للتصنيف
X = data[['ApplicantIncome', 'LoanAmount']].fillna(0)  # نعالج القيم الفارغة بـ 0
y = data['Loan_Status']

# تحويل التصنيفات إلى أرقام (لأن DecisionTree يتعامل مع أرقام)
y = y.map({'Y': 1, 'N': 0})

# تدريب نموذج Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# تصميم واجهة Streamlit
st.title('Loan Risk Prediction App')

st.write('أدخل المعلومات التالية لمعرفة إذا القرض خطير أو آمن')

# مدخلات المستخدم
income = st.number_input('دخل مقدم الطلب (Applicant Income)', min_value=0)
loan_amount = st.number_input('مبلغ القرض المطلوب (Loan Amount)', min_value=0)

# زر توقع
if st.button('توقع نتيجة القرض'):
    # نعمل التوقع
    prediction = model.predict([[income, loan_amount]])
    
    # نعرض النتيجة
    if prediction[0] == 1:
        st.success('القرض: منخفض الخطورة (تمت الموافقة عليه)')
    else:
        st.error('القرض: عالي الخطورة (تم رفضه)')