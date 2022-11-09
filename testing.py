import streamlit as st
import pandas as pd
import numpy as np

trendDF=pd.read_csv('assets/NPS_with_trends.csv',dtype=str)

df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))



parks = trendDF.ParkName.unique()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# [theme]
base='dark'
# backgroundColor = 'BAE5F9'

st.title('Welcome to our US National Park Recommender System')

st.write('Please select a park you visited and the month you visited from the drop-down menus on the left. If there are specific activities you are in interested in, add them too.')

st.header('Here are the parks we recommend for you.')

st.dataframe(df)

mapDF = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(mapDF)

add_park_selectbox = st.sidebar.selectbox(
    'What Park did you visit?',
    (parks)
)

add_month_selectbox = st.sidebar.selectbox(
    'What month did you visit?',
    # ('March', 'June', 'October')
    (months)
)

add_activity_selectbox = st.sidebar.selectbox(
    'What do you want to do?',
    ('Hike', 'Bike', 'Cemetary Exploration')
)

add_similar_selectbox = st.sidebar.radio(
    'Do you want to visit a similar park?',
    ('Yes', 'No')
)


# if add_similar_selectbox == 'Yes':
#     st.write('You want to visit a similar park.')
# else:
#     st.write('You want to visit a different type of park.')