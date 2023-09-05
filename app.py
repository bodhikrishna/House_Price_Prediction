import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
import folium
from matplotlib import pyplot as plt
import matplotlib
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

model_name = "House_price_model.pickle"
with open(model_name, "rb") as f:
    model = pickle.load(f)

with open("./columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]


def getmap(loc):
    proxy = "http://localhost:8888"
    geolocator = Nominatim(user_agent="MyApplication",proxies={"http":proxy})

    locationparam = geolocator.geocode(loc)
    lat = locationparam.latitude
    lng = locationparam.longitude
    return lat, lng


def predictPrice(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    prediction = model.predict([x])[0]
    return prediction


# main function


    
st.title("House Price Prediction")
st.markdown("Enter the details of the house to predict the price.")
# with open("style.css") as customCSS:
#     st.markdown(f"<style>{customCSS.read()}</style>", unsafe_allow_html=True)
# User inputs
location = st.selectbox("Select Desired Location", locations)


sqft = st.number_input("Sqft", value=0)
bhk = st.number_input("BHK")
bathrooms = st.number_input("No. of Bathrooms")

# making a prediction
if st.button("Predict"):
    predicted_price = predictPrice(location, sqft, bhk, bathrooms)
    st.success(f"Predicted Price: {predicted_price:.2f} lakhs")

    # loading Map
    latitude, longitude = getmap(location)

    m = folium.Map(location=[latitude, longitude], zoom_start=16)
    folium.Marker(
        [latitude, longitude],
        popup=f"{location}",
        tooltip="Click here to view location",
    ).add_to(m)
    # ESRI satellite level imagery
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        max_zoom=18,
    ).add_to(m)
    
    #layercontrol
    folium.LayerControl().add_to(m)
    
    st.subheader("Location of The Place")
    iframe = m._repr_html_()
    st.components.v1.html(iframe, height=500)
    
    
    st.subheader("Price vs. Total Sqft Distribution")
    fig=plt.figure(figsize=(8,6))
    hpd=pd.read_csv('house_metrics.csv')
    house_metrics= hpd.head(500)
    plt.scatter(house_metrics["total_sqft"], house_metrics["price"], color='green',s=20,alpha=0.9)
    plt.xlabel("total_sqft")
    plt.ylabel("price in lakhs")
    plt.xlim(0,4000)
    plt.ylim(0,500)
    plt.title("Price vs. Sqft")
    st.pyplot(fig)
    
    
    
    
    
    
     
    
    
    
