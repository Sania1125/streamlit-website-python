# streamlit_app.py


import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="NYC Taxi Data Explorer", layout="centered")

# Define layout sections
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# App title
st.title("🚖 NYC Taxi Data Dashboard")

# Header Section
with header:
    st.header("Welcome to the NYC Taxi Data Science Project 🚕")
    st.subheader("A mini dashboard powered by Streamlit and Python")
    user_input = st.text_area("🔍 Share your thoughts or observations:")
    st.write("💬 You typed:", user_input)
    st.markdown("This dashboard lets you explore pickup patterns in New York City using real-world Uber data.")

# Dataset Section
with dataset:
    st.header("🗂️ Dataset Overview")
    st.markdown("We are working with a sample of **Uber pickups in NYC (Sept 2014)**.")

    DATE_COLUMN = 'date/time'
    DATA_URL = 'https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz'

    @st.cache_data
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        data.columns = [col.lower() for col in data.columns]
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text('📥 Loading data...')
    data = load_data(10000)
    data_load_state.text('✅ Data loaded successfully!')

    if st.checkbox("Show raw data"):
        st.subheader("📝 Raw Data")
        st.write(data)

    # Plot hourly pickups
    st.subheader("📊 Number of pickups by hour")
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

    # Interactive map by hour
    hour_to_filter = st.slider("⏰ Select pickup hour", 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f"🗺️ Map of pickups at {hour_to_filter}:00")
    st.map(filtered_data)

# Features Section
with features:
    st.header("🔍 Custom Features")
    st.markdown("""
    - Cleaned and renamed columns
    - Hourly pickup visualization
    - Interactive map filter
    - User text input and feedback
    """)
    st.success("✨ New features added successfully!")
    st.balloons()

# Model Training Placeholder
with model_training:
    st.header("🤖 Model Training Section (Coming Soon)")
    st.markdown("Stay tuned! In future updates, this section will include ML models for predicting pickup demand.")




