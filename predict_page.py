import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_model():
    with open('predict_unicorn_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
enc_country = data["enc_country"]
enc_industry = data["enc_industry"]

def show_predict_page():
    st.title("Predict Unicorn Company")

    st.write("""Enter the country and industry of the company to predict its likelihood of becoming a Unicorn.""")
    
    st.sidebar.title("About")
    st.sidebar.info("This app predicts the likelihood of a company becoming a Unicorn based on its country and industry.")

    st.sidebar.title("Instructions")
    st.sidebar.markdown("1. Select the country of the company from the dropdown menu.")
    st.sidebar.markdown("2. Select the industry of the company from the dropdown menu.")
    st.sidebar.markdown("3. Click on the 'Predict' button to see the prediction.")

    st.sidebar.title("Developed by")
    st.sidebar.info(
        """
        Hewa Alegodage Nidula Chithwara
        """
    )

    st.subheader("Input Information")
    countries = ("China", "United States", "Germany", "India", "Canada",
                 "Switzerland", "Brazil", "United Kingdom", "Singapore", "Bermuda",
                 "Israel", "France", "The Netherlands", "Australia", "Finland",
                 "Indonesia", "Ireland", "Hong Kong", "Thailand", "Norway", "Chile",
                 "Malta", "Austria", "Mexico", "Estonia", "Spain", "Malaysia", "Liechtenstein",
                 "California", "Japan", "Belgium", "Poland", "Turkey", "United Arab Emirates",
                 "Sweden", "Portugal", "Cayman Islands", "South Korea", "Colombia", "Nigeria",
                 "Ecuador", "Denmark", "Egypt", "Vietnam", "South Africa", "Philippines",
                 "Lithuania", "Luxembourg", "Taiwan", "Croatia", "Czech Republic", "Italy",
                 "Seychelles", "Saudi Arabia", "Argentina", "Greece", "Senegal")

    industries = ("Enterprise Tech", "Media & Entertainment", "Industrials",
                  "Healthcare & Life Sciences", "Financial Services",
                  "Consumer & Retail", "Insurance", "Food and Beverage",
                  "Transportation", "Education", "Agricultural")

    country = st.selectbox("Country", countries)
    industry = st.selectbox("Industry", industries)

    ok = st.button("Predict")
    if ok:
        unseen_data = np.array([[country, industry]])
        encoded_country = enc_country.transform(unseen_data[:, 0].reshape(-1, 1)).toarray()
        encoded_industry = enc_industry.transform(unseen_data[:, 1].reshape(-1, 1)).toarray()
        prediction = model_loaded.predict(np.concatenate((encoded_country, encoded_industry), axis=1))

        # Mapping prediction values to categories
        prediction_category = ""
        if prediction[0] == 0:
            prediction_category = "Low"
        elif prediction[0] == 1:
            prediction_category = "Medium"
        elif prediction[0] == 2:
            prediction_category = "High"
    
        st.subheader(f"The company has a {prediction_category} chance of being a Unicorn")


if __name__ == "__main__":
    show_predict_page()
