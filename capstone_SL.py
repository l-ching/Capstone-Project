import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
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

import googlemaps


@st.cache(ttl=600)
def load_species():
    species = pd.read_csv('assets/all_species_112222.csv')
    species.drop(species.columns[species.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    return species

species = load_species()

@st.cache(ttl=600)
def load_locations():
    locations = pd.read_csv('assets/locations_112222.csv')
    locations.drop(locations.columns[locations.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    return locations

locations = load_locations()

@st.cache(ttl=600)
def load_weather():
    weather = pd.read_csv('assets/NPS_weather_trends_112222.csv')
    return weather

weather = load_weather()


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
base='dark'


###Intro text###

st.markdown("<h1 style='text-align: center; color: #213A1B;'>Welcome to our US National Park Recommender System</h1>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center; color: #213A1B;'>Please select a park you visited and the month you visited from the drop-down menus on the left. If there are specific activities you are in interested in, add them too</h3>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #213A1B;'>------------------------------------------------------------- </h2>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #213A1B;'>Below are the parks we recommend based on your inputs</h5>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #213A1B;'>The parks with all three activities matched are highlighted in blue</p>", unsafe_allow_html=True)

###Sidebar creation###

add_park_selectbox = st.sidebar.selectbox(
    'What park have you visited in the past?',
    (parks)
)

add_month_selectbox = st.sidebar.selectbox(
    'What month did you visit?',
    (months)
)

select_activities = st.sidebar.multiselect(
    'What are the top three activities you want to do?',
    (all_activities), max_selections = 3
)

add_similar_selectbox = st.sidebar.radio(
    'Do you want to visit a similar park?',
    ("Yes", "I'll try something new")
)

add_popular_selectbox = st.sidebar.radio(
    'Do you want to avoid crowds?',
    ("I don't mind crowds", "Yes")
)


city = st.sidebar.text_input('What city will you be traveling from?')
state = st.sidebar.text_input('What state will you be traveling from?')


###Create dataframe based off of month, park, and similar/different choices###
@st.cache

def cluster(park, month, sim_or_diff):

    # map user month to 
    '''Input user park and month, return silhouette score, map, and parks in cluster'''
    # filter for 10 yr avg
    
    park_weather = weather[['ParkName', 'Month', 'Year', 'Temp_Avg_Fahrenheit', 'Prcp_Avg_Inches']]

    # get 10-year averages for each month for each park
    avg_10_yr = park_weather.groupby(['ParkName','Month']).agg('mean').reset_index().drop(columns = 'Year')
    avg_10_yr.rename(columns = {'ParkName':'park', 'Month':'month','Temp_Avg_Fahrenheit':'temp', 'Prcp_Avg_Inches':'prcp'}, inplace = True)
    avg_10_yr['month_name'] = avg_10_yr['month'].apply(lambda x: calendar.month_name[x])

    avg_10_yr = avg_10_yr[avg_10_yr['month_name'] == month]
    
    clus_temp = avg_10_yr[['temp', 'prcp']]
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
    visualizer = KElbowVisualizer(KMeans(random_state = 42), k=(2,11), show = False)
    visualizer.fit(weather_sp_arr)
    optimal_k = visualizer.elbow_value_
    
    KM = KMeans(n_clusters = optimal_k, random_state = 42)
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
    if sim_or_diff == 'Yes':
        user_parks = temp_merged[temp_merged['k_cluster'] == user_cluster]['park'].tolist()
    elif sim_or_diff == "I'll try something new":
        user_parks = temp_merged[temp_merged['k_cluster'] != user_cluster]['park'].tolist()
    return user_parks


###Sort by popularity/crowdedness of parks###

def attendance_filter(park_list, popularity, sort_by_attendance = False, drop_percentage = 0.33):
    park_order_dict = {'Great Smoky Mountains National Park': 1,'Grand Canyon National Park': 2,'Yosemite National Park': 3,'Yellowstone National Park': 4,'Rocky Mountain National Park': 5,'Zion National Park': 6,'Olympic National Park': 7,'Grand Teton National Park': 8,'Acadia National Park': 9,'Cuyahoga Valley National Park': 10,'Glacier National Park': 11,'Indiana Dunes National Park': 12,'Joshua Tree National Park': 13,'Bryce Canyon National Park': 14,'Hawaii Volcanoes National Park': 15,'Hot Springs National Park': 16,'Shenandoah National Park': 17,'Mount Rainier National Park': 18,'Arches National Park': 19,'New River Gorge National Park and Preserve': 20,'Haleakala National Park': 21,'Death Valley National Park': 22,'Sequoia National Park': 23,'Everglades National Park': 24,'Badlands National Park': 25,'Capitol Reef National Park': 26,'Saguaro National Park': 27,'Petrified Forest National Park': 28,'Theodore Roosevelt National Park': 29,'Mammoth Cave National Park': 30,'Wind Cave National Park': 31,'Kings Canyon National Park': 32,'Canyonlands National Park': 33,'Crater Lake National Park': 34,'Biscayne National Park': 35,'Mesa Verde National Park': 36,'White Sands National Park': 37,'Denali National Park': 38,'Glacier Bay National Park': 39,'Lassen Volcanic National Park': 40,'Redwood National Park': 41,'Virgin Islands National Park': 42,'Carlsbad Caverns National Park': 43,'Big Bend National Park': 44,'Great Sand Dunes National Park': 45,'Channel Islands National Park': 46,'Kenai Fjords National Park': 47,'Voyageurs National Park': 48,'Black Canyon of the Gunnison National Park': 49,'Pinnacles National Park': 50,'Guadalupe Mountains National Park': 51,'Congaree National Park': 52,'Great Basin National Park': 53,'Wrangell - St Elias National Park': 54,'Dry Tortugas National Park': 55,'Katmai National Park': 56,'North Cascades National Park': 57,'Isle Royale National Park': 58,'National Park of American Samoa': 59,'Lake Clark National Park': 60,'Gates Of The Arctic National Park': 61,'Kobuk Valley National Park': 62}
    less_busy_parks ={}
    for key, value in park_order_dict.items():
        if value > 15:
            less_busy_parks[key] = value
    less_busy_parks_list = list(less_busy_parks.keys())
 
    full_park_list = list(park_order_dict.keys())

    if popularity == "I don't mind crowds":
        ordered_input_parks = full_park_list
    edit_list = []
    if popularity == 'Yes':
        for park in park_list:
            if park in less_busy_parks_list:
                edit_list.append(park)
        ordered_input_parks = edit_list
        
    drop_n_parks = int(len(park_list) * drop_percentage) + 1
    ordered_input_parks = ordered_input_parks[:len(ordered_input_parks)-drop_n_parks]

    if sort_by_attendance:
        return ordered_input_parks
    else:
        park_list = [park for park in park_list if park in ordered_input_parks]
        return park_list

# highlight most similar parks

def park_col(counter):
    if counter['score'] == 3:
        return 'blue'
    else:
        return 'gray'

    

def highlight_col(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #set by condition
    mask = df['Activity Matches'] == 3
    df.loc[mask, :] = 'background-color: lightblue'
    df.loc[~mask,:] = 'background-color: white'
    return df    


###activities filter to return final dataframe and map###


def activities_filter(lst_activities, lst_cluster, park):
    '''Input user activities, results from cluster fxn, user park, return final park map, activities df'''
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
    
    act_df.drop(columns = 'park', inplace = True)
    act_df.rename(columns = {'score':'Activity Matches'}, inplace = True)
    act_df['Park Name'] = act_df['Park Name'].str.replace('National Park', 'NP')
    
    act_df.index = np.arange(1, len(act_df)+1)
    
    
    ### order activities where user activities are first in list 
    for i in range(len(act_df)):
        park_activities = act_df['Activities'].iloc[i]
        for act in lst_activities:
            if act in park_activities:
                park_activities.insert(0, act)

        act_df['Activities'].iloc[i] = list(dict.fromkeys(park_activities))
    act_df = act_df[['Park Name', 'Activity Matches', 'Activities']]
    parks_return_df = act_df.copy()
    
    act_df = act_df.style.apply(highlight_col, axis = None)
    
    
    st.dataframe(act_df)
    st_folium(result_map, width = 725)
    
    return act_df, parks_return_df, result_map


lst_activities = select_activities
clus_results = cluster(add_park_selectbox, add_month_selectbox, add_similar_selectbox)

pop_results = attendance_filter(clus_results, add_popular_selectbox, sort_by_attendance = False, drop_percentage=0.33)


final = activities_filter(lst_activities, pop_results, add_park_selectbox)

parkList = final[1]
parkList = list(parkList['Park Name'].unique())


key = st.secrets['api_key']['key']

user_input = city + ',' + state


def drive_times(top_5_parks, user_input= user_input):
    gm = googlemaps.Client(key=key)
    api_call = gm.distance_matrix(origins=user_input, destinations = top_5_parks, mode = 'driving', units='imperial')
    time_list = []
    for i in range(5):
        try:
            time_list.append(api_call['rows'][0]['elements'][i]['duration']['text'])
        except:
            time_list.append('')

    if time_list == ['', '', '', '', '']:
        return st.write('Error in getting drive times, try a different input city')
    else:
        return time_list
    

# get top_5_parks from clustering, user input from streamlit
drive = drive_times(parkList, user_input)

driveDF = pd.DataFrame()
driveDF['Park Name'] = parkList
driveDF['Drive Time'] = drive
driveDF['Drive Time'].replace("", "drive unavailable", inplace=True)
driveDF.index = np.arange(1, len(driveDF)+1)

st.table(driveDF)
