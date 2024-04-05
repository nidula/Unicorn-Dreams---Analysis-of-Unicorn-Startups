import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Function to load the ML model
def load_model():
    with open('predict_unicorn_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load the model and encoders
data = load_model()
model_loaded = data["model"]
enc_country = data["enc_country"]
enc_industry = data["enc_industry"]

# Add CSS styling to remove padding and margins around the logo
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding: 0px;
        margin: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display the prediction page
def show_predict_page():
    st.title("Predicting the Likelihood of a Business becoming a Unicorn")
    st.write("""Enter the country and industry in which you have a business or are planning to establish to see the results.""")
    st.sidebar.image("logo.png")  # Add the same logo here
    st.sidebar.title("About")
    st.sidebar.info("This machine learning model predicts the likelihood of a company becoming a Unicorn based on the country and industry it is located in since we assume those factors play a vital role in the growth of any business.")
    st.sidebar.title("Instructions")
    st.sidebar.markdown("1. Select the country.")
    st.sidebar.markdown("2. Select the industry of the business.")
    st.sidebar.markdown("3. Click on “Predict” to see the results.")
    st.sidebar.title("Group Members")
    st.sidebar.info(
        """
        Hewa Alegodage Nidula Chithwara\n
        Harsimranjit Kaur\n
        Noufia Najeeb\n
        Elson Jacob\n
        Shraddhaba Bharatsinh Jadeja 
        """
        )


    st.subheader("Business Information")
    countries = ("Argentina", "Australia", "Austria", "Belgium", "Bermuda","Brazil", "California", "Canada", "Cayman Islands", 
                 "Chile","China", "Colombia", "Croatia", "Czech Republic", "Denmark","Ecuador", "Egypt", "Estonia", "Finland", 
                 "France","Germany", "Greece", "Hong Kong", "India", "Indonesia","Ireland", "Israel", "Italy", "Japan", "Liechtenstein",
                 "Lithuania", "Luxembourg", "Malaysia", "Malta", "Mexico","Netherlands", "Nigeria", "Norway", "Philippines", "Poland",
                 "Portugal", "Saudi Arabia", "Senegal", "Seychelles", "Singapore","South Africa", "South Korea", "Spain", "Sweden", 
                 "Switzerland","Taiwan", "Thailand", "Turkey", "United Arab Emirates", "United Kingdom","United States", "Vietnam")

    industries = ("Agricultural", "Consumer & Retail", "Education", "Enterprise Tech", "Financial Services", "Food and Beverage", 
                  "Healthcare & Life Sciences", "Industrials", "Insurance", "Media & Entertainment", "Transportation")

    country = st.selectbox("Country of Business", countries)
    industry = st.selectbox("Industry of Business", industries)

    ok = st.button("Predict")
    if ok:
        unseen_data = np.array([[country, industry]])
        encoded_country = enc_country.transform(unseen_data[:, 0].reshape(-1, 1)).toarray()
        encoded_industry = enc_industry.transform(unseen_data[:, 1].reshape(-1, 1)).toarray()
        prediction = model_loaded.predict(np.concatenate((encoded_country, encoded_industry), axis=1))

        # Mapping prediction values to categories
        prediction_category = ""
        if prediction[0] == 0:
            prediction_category = "Low Likelihood"
        elif prediction[0] == 1:
            prediction_category = "Medium likelihood"
        elif prediction[0] == 2:
            prediction_category = "High likelihood"
    
        st.write(f"Based on the information provided, the business has a <span style='font-size:40px'>{prediction_category}</span> of becoming a Unicorn according to current market scenarios.", unsafe_allow_html=True)

if __name__ == "__main__":
    show_predict_page()
