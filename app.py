import streamlit as st
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

loaded_model = tf.keras.models.load_model('./Models/model_3_aug')

st.title('Dog Vs. Cat Classifier : ')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = plt.imread(uploaded_file)
    img = cv.resize(img,(128,128))
    plt.imshow(img)
    Model_Prediction = loaded_model.predict(tf.expand_dims(img, axis=0), verbose=0)[0][0]
    st.subheader('Prediction')
    if Model_Prediction == 1:
        st.title('Dog')
    else:
        st.title('Cat')