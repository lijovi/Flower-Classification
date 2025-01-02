import streamlit as st 
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.title("Flower Classification")

st.write("Flower Categories :")
col = st.columns(5)
category = ["Bluebell", "Buttercup", "Cowslip", "Crocus", "Daffodil", "Daisy", "Dandelion", "Fritillary", "Iris", "Pansy", "Rose" ,"Snowdrop", "Sunflower", "Tiger Lily", "Tulip", "Windflower"]
for i in range(4):
    col[0].write(category[i])
for i in range(4, 7):
    col[1].write(category[i])
for i in range(7, 10):
    col[2].write(category[i])
for i in range(10, 13):
    col[3].write(category[i])
for i in range(13, 16):
    col[4].write(category[i])

def preprocess_input_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0
    return img_array

def predict_flower(img_path, model):
    img_array = preprocess_input_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    flower_name = category[predicted_class_idx]
    return flower_name

with open('./model_inception.h5', 'rb') as f:
    model = pickle.load(f)

input = st.file_uploader("Choose a flower", type=["png", "jpg", "jpeg"])
if input:
    _, center, _ = st.columns(3)
    with center:
        st.image(input, caption="Flower", width=300)
        pred = predict_flower(input, model)
        st.write(f"Flower Category is {pred}")

# py -m streamlit run Flower_Classification.py