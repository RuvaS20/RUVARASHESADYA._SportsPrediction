import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model, encoder, and scaler
model = pickle.load(open('Best_XGBRegressor.pkl', 'rb'))
preferred_foot_encoder = pickle.load(open('preferred_foot_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main(): 
        # Set page background color
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FFF8F0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # st.title("Football Player Overall Score Predictor")
    html_temp = """
    <div style="background:#111D4A ;padding:10px">
    <h2 style="color:#FFF8F0;text-align:center;">Player Overall Score Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    potential = st.slider("Potential", 0.0, 100.0, 50.0)
    value_eur = st.number_input("Value (EUR)", 0.0)
    wage_eur = st.number_input("Wage (EUR)", 0.0)
    international_reputation = st.slider("International Reputation", 1, 5, 1)
    passing = st.slider("Passing", 0.0, 100.0, 50.0)
    movement_reactions = st.slider("Movement Reactions", 0.0, 100.0, 50.0)
    preferred_foot = st.selectbox("Preferred Foot", ["Left", "Right"])
    
    if st.button("Predict"): 
        # Create a dataframe with the input
        data = {
            'potential': potential,
            'value_eur': value_eur,
            'wage_eur': wage_eur,
            'international_reputation': international_reputation,
            'passing': passing,
            'movement_reactions': movement_reactions,
            'preferred_foot': preferred_foot
        }
        df = pd.DataFrame([data])
        
        # Encode the 'preferred_foot' column
        df['preferred_foot'] = preferred_foot_encoder.transform(df['preferred_foot'])
        
        # Scale the input data
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)
        
        output = int(prediction[0])
        
        st.success(f'Predicted Overall Score: {output}')
      
if __name__=='__main__': 
    main()
