import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('predict_unicorn_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
encoded_country = data["encoded_country"]
encoded_industry = data["encoded_industry"]