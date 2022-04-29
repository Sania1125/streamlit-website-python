import streamlit as st

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.title('FirstApp')

with header:
    st.title('Welcome to Awesome Data Science Project')
    st.subheader('My very own subheader is this one!')
    st.text_area(label="Give some text here: ")
    st.text('This is just ordinary text in normal font and size. Quick brown fox jumps over the lazy dog.')

with dataset:
    st.title('NYC Taxi Dataset')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

import numpy as np
import pandas as pd


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)


with features:
    st.header("The Featuers")
    st.balloons()
    st.text('Here are the featuers which I created recently. An as thy say, the devil is always in the featuers!!')
    
with modelTraining:
    st.title('Model Training')
