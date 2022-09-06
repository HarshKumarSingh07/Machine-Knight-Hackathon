import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from utils import *
import pickle
df = pd.read_csv("train.csv")
num_cols = check_numerical(df)
categorical_cols = check_categorical(df)

final_cols = num_cols+categorical_cols

X = df[final_cols].copy()
y = df.rent

st.write("""
# MACHINEKNIGHT HACKATHON
This web-app predicts the **Real Estate rent prices**
Harsh Kumar Singh Team - AI-HUSTLERS!
""")
st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Continuos Input Values')

def user_input_features():
    lat = st.sidebar.slider('Latitude', float(X.latitude.min()), float(X.latitude.max()), float(X.latitude.mean()),format="%f")
    longi = st.sidebar.slider('Longitude', float(X.longitude.min()), float(X.longitude.max()), float(X.longitude.mean()), format="%f")
    gym = st.sidebar.slider('Gym', float(X.gym.min()), float(X.gym.max()), float(X.gym.mean()),format="%f")
    lift = st.sidebar.slider('Lift', float(X.lift.min()), float(X.lift.max()), float(X.lift.mean()),format="%f")
    sp = st.sidebar.slider('Swimming Pool', float(X.swimming_pool.min()), float(X.swimming_pool.max()),format="%f")
    nego = st.sidebar.slider('Negotiable', float(X.negotiable.min()), float(X.negotiable.max()), float(X.negotiable.mean()),format="%f")
    ps = st.sidebar.slider('Property Size', float(X.property_size.min()), float(X.property_size.max()), float(X.property_size.mean()),format="%f")
    pa = st.sidebar.slider('Property Age', float(X.property_age.min()), float(X.property_age.max()), float(X.property_age.mean()),format="%f")
    bath = st.sidebar.slider('Bathroom', float(X.bathroom.min()), float(X.bathroom.max()), float(X.bathroom.mean()),format="%f")
    cup = st.sidebar.slider('Cupboard', float(X.cup_board.min()), float(X.cup_board.max()), float(X.cup_board.mean()),format="%f")
    floor = st.sidebar.slider('Floor', float(X.floor.min()), float(X.floor.max()), float(X.floor.mean()),format="%f")
    tfloor = st.sidebar.slider('Total Floors', float(X.total_floor.min()), float(X.total_floor.max()), float(X.total_floor.mean()),format="%f")
    balc = st.sidebar.slider('Balconies', float(X.balconies.min()), float(X.balconies.max()), float(X.balconies.mean()),format="%f")
    
    typ = st.text_input("Type : ")
    lt = st.text_input("Lease Type : ")
    fur = st.text_input("Furnishing : ")
    park = st.text_input("Parking : ")
    fac = st.text_input("Facing : ")
    ws = st.text_input("Water Supply : ")
    bt = st.text_input("Building Type")
    amen = st.text_input("Amenities")
    loc = st.text_input("Locality")
    act = st.text_input("Activation Date")
    data = {
        "latitude":lat,
        "longitude":longi,
        "gym":gym,
        "lift":lift,
        "swimming_pool":sp,
        "negotiable":nego,
        "property_size":ps,
        "property_age":pa,
        "bathroom":bath,
        "cupboard":cup,
        "floor":floor,
        "total_floor":tfloor,
        "balconies":balc,
        "amenities":amen,
        "locality":loc,
        "activation_date":act,
        "type":typ,
        "lease_type":lt,
        "furnishing":fur,
        "parking":park,
        "facing":fac,
        "water_supply":ws,
        "building_type":bt

    }
    features = pd.DataFrame(data, index=[0])
    return features

dataframe = user_input_features()

numeric_data = check_numerical(dataframe)
categorical_data = check_categorical(dataframe)
my_cols = numeric_data + categorical_data
dataframe = dataframe[my_cols].copy()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
with open('pickle_model', 'rb') as f:
    model = pickle.load(f)

# Apply Model to Make Prediction

prediction = model.predict(dataframe)

st.header('Prediction of RENT')
st.write(prediction)
st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')