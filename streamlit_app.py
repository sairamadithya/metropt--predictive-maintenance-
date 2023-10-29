#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import lime
from lime import lime_tabular
html_temp = """ 
  <div style="background-color:orange ;padding:7px">
  <h2 style="color:black;text-align:center;"><b>DASHBOARD FOR METRO TRAIN ANALYSIS USING MACHINE LEARNING<b></h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
model=tf.keras.models.load_model(r"metropt multitask ann model-1.h5")
file=st.file_uploader('upload the sensor data file',type=['csv','xlsx'])
if file is not None:
    data=pd.read_csv(file)
    prediction = model.predict(data)
    p1= np.argmax(prediction[0],axis=1)
    p2= np.argmax(prediction[1],axis=1)
    p3= np.argmax(prediction[2],axis=1)
    a=data.loc[:,['TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Flowmeter','Motor_current']]
    labels_a=['TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Flowmeter','Motor_current']
    b=data.loc[:,['COMP','DV_eletric','Towers','MPG','LPS','Pressure_switch','Oil_level','Caudal_impulses']]
    labels_b=['COMP','DV_eletric','Towers','MPG','LPS','Pressure_switch','Oil_level','Caudal_impulses']
    c=data.loc[:,['gpsLong','gpsLat','gpsSpeed']]
    c.rename(columns = {'gpsLong':'longitude'}, inplace = True)
    c.rename(columns = {'gpsLat':'latitude'}, inplace = True)
    x1,_,_,x3 = st.columns(4)
    x1= st.area_chart(a)
    x3=st.bar_chart(b)
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.map(c)
    nn=st.slider('select the data for prediction',0,100,1)
    if p1[nn]==0:
                    st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure status- Air failure</h2>
  </div>
  """ ,unsafe_allow_html=True)
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
    elif p1[nn]==1:
                    st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure status- Oil failure</h2>
  </div>
  """ ,unsafe_allow_html=True)
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
    elif p1[nn]==2:
        st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure status- No failure</h2>
  </div>
  """ ,unsafe_allow_html=True)
        st.write('  ')
        st.write('  ')
        st.write('  ')
        st.write('  ')
    if p2[nn]==0:
                st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure location- Air dryer</h2>
  </div>
  """ ,unsafe_allow_html=True)
                st.write('  ')
                st.write('  ')
                st.write('  ')
                st.write('  ')
    elif p2[nn]==1:
                st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure location- Client</h2>
  </div>
  """ ,unsafe_allow_html=True)  
                st.write('  ')
                st.write('  ')
                st.write('  ')
                st.write('  ')
    elif p2[nn]==2:
                st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure location- Compressor</h2>
  </div>
  """ ,unsafe_allow_html=True)
                st.write('  ')
                st.write('  ')
                st.write('  ')
                st.write('  ')
    elif p2[nn]==3: 
                st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Failure location- No location</h2>
  </div>
  """ ,unsafe_allow_html=True)
                st.write('  ')
                st.write('  ')
                st.write('  ')
                st.write('  ')
    if p3[nn]==0:
                st.markdown(""" 
  <div style="background-color: red;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">GPS Quality- Bad</h2>
  </div>
  """ ,unsafe_allow_html=True)
                st.write('  ')
                st.write('  ')
                st.write('  ')
                st.write('  ')
    else:
                    st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">GPS Quality- Good</h2>
  </div>
  """ ,unsafe_allow_html=True)
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
                    st.write('  ')
    st.subheader('LIME Explanations')
    explainer_f = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns,
                                                      class_names=['air failure','oil failure','normal'], mode='classification')
    explainer_l = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns,
                                                      class_names=['air dryer','client','compressor','normal'], verbose=True, mode='classification')
    explainer_g = lime.lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns,
                                                      class_names=['good','bad'], verbose=True, mode='classification')
    def f(X):
        a1=model.predict(data)
        return a1[0]
    def l(X):
        a1=model.predict(data)
        return a1[1]
    def g(X):
        a1=model.predict(data)
        return a1[2]
    j=st.slider('enter the data you want to analyze',0,100,1)
    exp_f = explainer_f.explain_instance(data.values[j], f,num_samples=100,num_features=24)
    exp_l = explainer_l.explain_instance(data.values[j], l,num_samples=100,num_features=24)
    exp_g = explainer_g.explain_instance(data.values[j], g,num_samples=100,num_features=24)
    st.subheader('LIME Explanations for Fault type')
    components.html(exp_f.as_html(), height=1000)
    st.subheader('LIME Explanations for Fault location')
    components.html(exp_l.as_html(), height=1000)
    st.subheader('LIME Explanations for GPS Quality')
    components.html(exp_g.as_html(), height=1000)

st.write('Developed by Sairam.V.A And Pratik Jadhav')
st.write(' ')
st.write('https://linktr.ee/sairamadithya')
st.write(' ')
st.write('https://github.com/pratik0502')

