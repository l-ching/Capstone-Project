import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium

trendDF=pd.read_csv('assets/NPS_with_trends.csv',dtype=str)

parks = trendDF.ParkName.unique()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# [theme]
base='dark'
# backgroundColor = 'BAE5F9'

st.title('Welcome to our US National Park Recommender System')

st.write('Please select a park you visited and the month you visited from the drop-down menus on the left. If there are specific activities you are in interested in, add them too.')

st.header('Here are the parks we recommend for you.')


###Sidebar creation


add_park_selectbox = st.sidebar.selectbox(
    'What Park did you visit?',
    (parks)
)

add_month_selectbox = st.sidebar.selectbox(
    'What month did you visit?',
    # ('March', 'June', 'October')
    (months)
)

add_activity1_selectbox = st.sidebar.selectbox(
    'What is your first most important activity?',
    ('Hike', 'Bike', 'Cemetary Exploration')
)

add_activity2_selectbox = st.sidebar.selectbox(
    'What is your second most important activity?',
    ('Hike', 'Bike', 'Cemetary Exploration')
)

add_activity3_selectbox = st.sidebar.selectbox(
    'What is your third most important activity?',
    ('Hike', 'Bike', 'Cemetary Exploration')
)

add_similar_selectbox = st.sidebar.radio(
    'Do you want to visit a similar park?',
    ('Yes', 'No')
)



###Display DF and Map creation

locDF = pd.read_csv('./assets/parks_update.csv',dtype=str)

location_list_yes = locDF[['Latitude', 'Longitude']].values.tolist()[0:5]

location_list_no = locDF[['Latitude', 'Longitude']].values.tolist()[40:45]

park_names_yes = locDF['Park Name'].tolist()[0:5]
pnyDF = pd.DataFrame(park_names_yes, columns = ['Park Name'])


park_names_no = locDF['Park Name'].tolist()[40:45]
pnnDF = pd.DataFrame(park_names_no, columns = ['Park Name'])

us_map_yes = folium.Map(location=[48, -102], zoom_start=3)
us_map_no = folium.Map(location=[48, -102], zoom_start=3)

    
for point in range(0, len(location_list_yes)):
    folium.Marker(location_list_yes[point],popup=park_names_yes[point], icon=folium.Icon(color='blue', icon_color='white', icon='star', angle=0, prefix='fa')).add_to(us_map_yes)

for point in range(0, len(location_list_no)):
    folium.Marker(location_list_no[point],popup=park_names_no[point], icon=folium.Icon(color='blue', icon_color='white', icon='star', angle=0, prefix='fa')).add_to(us_map_no)
    
if add_similar_selectbox == 'Yes':
    st.dataframe(pnyDF)
    st_folium(us_map_yes, width = 725)
else:
    st.dataframe(pnnDF)
    st_folium(us_map_no, width = 725)






