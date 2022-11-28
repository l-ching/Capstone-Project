import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
# import matplotlib.pyplot as plt
import calendar
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from collections import Counter

import yellowbrick
from yellowbrick.cluster import KElbowVisualizer
import ast
pd.options.mode.chained_assignment = None



species = pd.read_csv('assets/all_species_112222.csv')
species.drop(species.columns[species.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

locations = pd.read_csv('assets/locations_112222.csv')
locations.drop(locations.columns[locations.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

weather = pd.read_csv('assets/NPS_weather_trends_112222.csv')

activities = pd.read_csv('assets/activities.csv')
activities['cleaned'] = [ast.literal_eval(activities['cleaned'].iloc[i]) for i in range(len(activities))]
all_activities = [x for sublist in activities.cleaned.tolist()for x in sublist]
all_activities = set(all_activities)
all_activities = sorted(all_activities)



def regioncolors(counter):
    if counter['k_cluster'] == 0:
        return 'darkblue'
    elif counter['k_cluster'] == 1:
        return 'darkred'
    elif counter['k_cluster'] == 2:
        return 'blue'
    elif counter['k_cluster'] == 3:
        return 'darkgreen'
    else:
        return 'darkpurple'
    



parks = weather.ParkName.unique()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# [theme]
base='dark'
# backgroundColor = 'BAE5F9'

###Intro text###

st.markdown("<h1 style='text-align: center; color: black;'>Welcome to our US National Park Recommender System</h1>", unsafe_allow_html=True)


st.markdown("<p style='text-align: center; color: black;'>Please select a park you visited and the month you visited from the drop-down menus on the left. If there are specific activities you are in interested in, add them too</p>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: black;'>Here are the parks we recommend for you.</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: black;'>The parks with all three activities matched are highlighted in blue</p>", unsafe_allow_html=True)

###Sidebar creation###


add_park_selectbox = st.sidebar.selectbox(
    'What Park did you visit?',
    (parks)
)


add_month_selectbox = st.sidebar.selectbox(
    'What month did you visit?',
    # ('March', 'June', 'October')
    (months)
)

# add_activity1_selectbox = st.sidebar.selectbox(
#     'What is your first most important activity?',
#     (all_activities)
# )

# add_activity2_selectbox = st.sidebar.selectbox(
#     'What is your second most important activity?',
#     (all_activities)
# )

# add_activity3_selectbox = st.sidebar.selectbox(
#     'What is your third most important activity?',
#     (all_activities)
# )


###multi-select instead of individual###


select_activities = st.sidebar.multiselect(
    'What are the top three activities you want to do?',
    (all_activities), max_selections = 3
)

add_similar_selectbox = st.sidebar.radio(
    'Do you want to visit a park similar in popularity?',
    ('Yes', 'No')
)




###Create dataframe based off of month, park, and similar/different choices###

def cluster(park, month, sim_or_diff):

    # map user month to 
    '''Input user park and month, return silhouette score, map, and parks in cluster'''
    # filter for 10 yr avg
    
    park_weather = weather[['ParkName', 'Month', 'Year', 'Temp_Avg_Fahrenheit', 'Prcp_Avg_Inches']]
    #park_weather = park_weather[park_weather['Year'] == 2021]

    # get 10-year averages for each month for each park
    avg_10_yr = park_weather.groupby(['ParkName','Month']).agg('mean').reset_index().drop(columns = 'Year')
    avg_10_yr.rename(columns = {'ParkName':'park', 'Month':'month','Temp_Avg_Fahrenheit':'temp', 'Prcp_Avg_Inches':'prcp'}, inplace = True)
    #avg_10_yr.drop(columns = 'prcp', inplace = True)
    avg_10_yr['month_name'] = avg_10_yr['month'].apply(lambda x: calendar.month_name[x])

    avg_10_yr = avg_10_yr[avg_10_yr['month_name'] == month]
    
    clus_temp = avg_10_yr[['temp', 'prcp']]
    # st.write(clus_temp)
    X = StandardScaler().fit_transform(clus_temp)

    # sort park names for future merging
    temp_merged = avg_10_yr.merge(locations, how = 'left', left_on = 'park', right_on = 'Park Name')
    sort_parks = temp_merged['Park Code'].tolist()
    # new df with only species that are present 
    present_sp = species[species['Occurrence'] == 'Present']
    similarity_df = present_sp[['Park Name', 'Scientific Name', 'Park Code']]
    sort_i = dict(zip(sort_parks, range(len(sort_parks))))
    similarity_df['park_code_ranked'] = similarity_df['Park Code'].map(sort_i)
    similarity_df.sort_values(by = ['park_code_ranked'], inplace = True)
    similarity_df.drop('park_code_ranked', axis = 1, inplace = True)

    # list of all park codes
    park_codes = list(similarity_df['Park Code'].unique())
    # list of all species in a park
    code_sp_list = []
    for code in park_codes:
        sp = list(similarity_df[similarity_df['Park Code'] == code]['Scientific Name'])
        #sp.append(code)
        code_sp_list.append(sp)

    # new park-species dataframe
    park_sp_df = pd.DataFrame()
    park_sp_df['park_code'] = park_codes
    park_sp_df['species_list'] = code_sp_list

    # 0 if species is not present, 1 if species is present
    mlb = MultiLabelBinarizer()
    vec = mlb.fit_transform(park_sp_df['species_list'])
    vecs = pd.DataFrame(vec, columns=mlb.classes_)

    # apply cosine_similarity fxn on df
    df_cosine = pd.DataFrame(cosine_similarity(vecs,dense_output=True))

    # pca for dimensitonality reduction
    pca = PCA(n_components =4)
    transform = pca.fit_transform(df_cosine)

    # concat weather and species vecs
    weather_sp = []
    for i in range(len(park_sp_df)):
        concat = np.concatenate((X[i], transform[i]))
        weather_sp.append(concat)
    weather_sp_arr = np.array(weather_sp)
    KM = KMeans(n_clusters = 5, random_state = 42)
    #model = KM.fit(X)
    temp_labels = KM.fit_predict(weather_sp_arr)

    labs = np.unique(temp_labels)
    avg_10_yr['k_cluster'] = temp_labels
    temp_merged = avg_10_yr.merge(locations, how = 'left', left_on = 'park', right_on = 'Park Name')

    location_list = temp_merged[['Latitude', 'Longitude']].values.tolist()
    park_names = temp_merged['Park Name'].tolist()

            
    temp_merged['color'] = temp_merged.apply(regioncolors, axis = 1)

    us_map = folium.Map(tiles='CartoDB positron', zoom_start=14)

    for point in range(0, len(location_list)):
        folium.Marker(location_list[point],popup=park_names[point], icon=folium.Icon(color=temp_merged["color"][point], icon_color='white', icon='star', angle=0, prefix='fa')).add_to(us_map)

    sil_score = silhouette_score(weather_sp_arr, KM.fit_predict(weather_sp_arr)) # a good silhouette score should be > 0.5
    # print('Silhouette Score:', sil_score)
    user_cluster = temp_merged[temp_merged['park'] == park]['k_cluster'].item()
    user_parks = temp_merged[temp_merged['k_cluster'] == user_cluster]['park'].tolist()
    # display(us_map)
    if sim_or_diff == 'Yes':
        user_parks = temp_merged[temp_merged['k_cluster'] == user_cluster]['park'].tolist()
        # st.dataframe(user_parks.head())
        # st_folium(us_map, width = 725)
    elif sim_or_diff == 'No':
        user_parks = temp_merged[temp_merged['k_cluster'] != user_cluster]['park'].tolist()
        # st.dataframe(user_parks.head())
        # st_folium(us_map, width = 725)
    # st.dataframe(user_parks)
    return user_parks



# highlight most similar parks

def park_col(counter):
    if counter['score'] == 3:
        return 'blue'
    elif counter['score'] == 2:
        return 'darkblue'
    elif counter['score'] == 1:
        return 'darkred'
    elif counter['score'] == 0:
        return 'darkgreen'
    

def highlight_col(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #set by condition
    mask = df['score'] == 3
    df.loc[mask, :] = 'background-color: lightblue'
    # df.loc[~mask,:] = 'background-color: ""'
    df.loc[~mask,:] = 'background-color: white'
    return df    


###activities filter to return final dataframe and map###

def activities_filter(lst_activities, lst_cluster, park):
    '''Input user activties, results from cluster fxn, user park, return final park map, activities df'''
    act_df = activities.merge(locations, how = 'left', left_on = 'park', right_on = 'Park Code')[['park', 'Park Name','cleaned']]
    user_acts = act_df[act_df['Park Name'].isin(lst_cluster)]
    # rank parks based on count of user activities 
    scores = []
    for i in range(len(user_acts)):
        if len(set(lst_activities).intersection(set(user_acts['cleaned'].iloc[i]))) == 3:
            scores.append(3)
        elif len(set(lst_activities).intersection(set(user_acts['cleaned'].iloc[i]))) == 2:
            scores.append(2)
        elif len(set(lst_activities).intersection(set(user_acts['cleaned'].iloc[i]))) == 1:
            scores.append(1)
        else:
            scores.append(0)
    
    user_acts['score'] = scores
    sorted_acts = user_acts.sort_values(by = 'score', ascending = False)
    # if user park is in final filtered df, drop row
    sorted_acts = sorted_acts[sorted_acts['Park Name'].str.contains(park) == False]

    act_df = sorted_acts.iloc[:5]
    act_df.rename(columns = {'cleaned':'Activities'},inplace= True)
    act_df.reset_index(inplace = True, drop = True)

    act_loc_df = act_df.merge(locations, left_on = 'Park Name', right_on = 'Park Name')
    act_loc_df['color'] = act_loc_df.apply(park_col, axis = 1)
    

    location_list = act_loc_df[['Latitude', 'Longitude']].values.tolist()
    park_names = act_loc_df['Park Name'].tolist()

    # display final user map and final activities df
    result_map = folium.Map(tiles='CartoDB positron', zoom_start=14)
    for point in range(0, len(location_list)):
        folium.Marker(location_list[point],popup=park_names[point], tooltip = park_names[point], icon=folium.Icon(color=act_loc_df["color"][point], icon_color='white', icon='star', angle=0, prefix='fa')).add_to(result_map)
    
    #act_df = act_df.style.apply(change_df_col, axis = 1)
    #return act_df[['Park Name', 'Activities']]
    act_df = act_df.style.apply(highlight_col, axis = None)
    
    st.dataframe(act_df)
    st_folium(result_map, width = 725)
    
    return act_df




lst_activities = select_activities
clus_results = cluster(add_park_selectbox, add_month_selectbox, add_similar_selectbox)

# rank on attendance?
activities_filter(lst_activities, clus_results, add_park_selectbox)