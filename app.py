import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load model once
@st.cache_resource
def load_trained_model():
    return load_model("mnist_model.h5")

model = load_trained_model()

# Load MNIST dataset (for real image sampling)
@st.cache_data
def load_data():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    return x_train, y_train

x_train, y_train = load_data()

# App layout
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    # Select 5 samples of the selected digit
    digit_imgs = x_train[y_train == digit]
    selected = digit_imgs[np.random.choice(len(digit_imgs), 5, replace=False)]

    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(selected[i], width=64, caption=f"Sample {i+1}")
