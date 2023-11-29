import pandas as pd
import numpy as np
from numpy import mean
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
from shap.plots import _waterfall
import lime
from lime import lime_tabular

plt.style.use('default')

# dashboard title
st.set_page_config(
    page_title = 'Real-Time Full Stomach Prediction',
    page_icon = ':cactus:',
    layout = 'wide'
)
st.markdown("<h1 style='text-align: center; color: black;'>Real-Time Full Stomach Prediction</h1>",
 unsafe_allow_html=True)
train = pd.read_csv('train.csv',encoding='utf-8')

x_train = train.drop(columns=['Outcome'],axis=1)
y_train = train['Outcome'].copy()


rf = RandomForestClassifier(n_estimators=500,min_samples_leaf=4,min_samples_split=4,max_depth=6,random_state=1)

rf.fit(x_train,y_train)


# side-bar 
def user_input_features():
    st.sidebar.header('Please input parameters below:')
    a1 = st.sidebar.number_input('Age(year)', min_value=18,max_value=85)
    a2 = st.sidebar.number_input('Fasting time(h)', min_value=8,max_value=24)
    a3 = st.sidebar.selectbox("Sex", ('Male', 'Female'))
    a4 = st.sidebar.selectbox("Diabetes", ('No','Yes'))
    a5 = st.sidebar.selectbox("GERD", ('No','Yes'))
    a6 = st.sidebar.selectbox("General food", ('No','Yes'))
    result = ""
    
    if a3 == 'Male':
        a3 = 0
    else:
        a3 = 1
    if a4 == 'Yes':
        a4 = 1
    else:
        a4 = 0
    if a5 == 'Yes':
        a5 = 1
    else:
        a5 = 0
    if a6 == 'Yes':
        a6 = 1
    else:
        a6 = 0
        
    output = [a1,a2,a3,a4,a5,a6]
    return output
outputdf = user_input_features()
outputdf = pd.DataFrame([outputdf],columns=x_train.columns)
p1 = rf.predict(outputdf)[0]
p2 = rf.predict_proba(outputdf)
p3 = p2[:,1]
#explainer = shap.TreeExplainer(rf)
#shap_values = explainer.shap_values(outputdf)
if float(p3) > 0.468:
    b = "Full stomach"
else:
   
    b = "Empty stomach"

if st.button("Predict"): 
    st.success('The probability of Full stomach: {:.2f}%'.format(p3[0]*100))
    st.success('The risk group'+ b)
 

    explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(trainx),
    feature_names=trainx.columns,
    class_names=['empty', 'full'],
    mode=“classification”)

    exp = explainer.explain_instance(data_row=np.squeeze(outputdf.T), predict_fn=Cb.predict_proba)
    exp.show_in_notebook(show_table=True)
    
    #st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #shap.summary_plot(shap_values,outputdf,feature_names=X.columns)


st.markdown("*Statement: this website will not record or store any information inputed.")
st.write("2022 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
st.write("✉ Contact Us: zoujianjun100@126.com")
