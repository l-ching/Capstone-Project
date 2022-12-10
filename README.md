# Capstone Project: National Park Recommender System
![Alt text](assets/20190727_073434.jpg?raw=true "Somewhere near Yosemite")
<sub>Photo of Yosemite National Park by Michael Kozma</sub>
### Authors:
- Lindsey Ching (lching)
- Amanda Davis (azdavis)
- Michael Kozma (mkozma)

## Introduction:
Like many people, the three of us have a passion for the outdoors, and we specifically love the national parks. Also like many other people, we haven’t had time to visit all 63 parks or even be familiar with all of them. That is why we decided to build this recommender engine; to assist with picking out a park which can suit the user's preferences on a variety of criteria. This README will expound upon the details of digging into the source code if you want to see how everything works under the hood. However, the best way to use the tool is to go to our hosted Streamlit app [here](https://l-ching-capstone-project-capstone-sl-th8bn9.streamlit.app/) and start playing around with it yourself. If you want to run it locally or peek under the hood though, read on!

## Dive in!
### Clone the repo:
```
git clone https://github.com/l-ching/Capstone-Project.git
```
### Then, install the necessary packages:
```
pip install -r requirements.txt
```

## Understanding the repo layout:
Most of the files in the repo are not actually used in running the Streamlit app. This is because most of the work of this project was done in cultivating the dataset, and the end app is using machine learning and the Streamlit front-end to mostly just facilitate exploring that dataset. Because of this, we have left in many files which aren't actively run for the app, but which give insight into how we cultivated the dataset that is used.

### Streamlit folder:
Holds the configuration file for Streamlit.
### Assets folder:
Holds all the data in various files written and used by different notebooks. All the data we used is publicly available. The dataset that we started with was publicly available national park attendance data which can be accessed [here](https://irma.nps.gov/STATS/), species data from the national park service [here](https://irma.nps.gov/NPSpecies/), and other species data from [here](https://www.kaggle.com/datasets/nationalparkservice/park-biodiversity?select=species.csv).
### Weather_and_trends folder:
Holds three notebooks for getting the google trends and weather data.
- `NPS_trends.ipynb`
Shows our collection of Google Trends data for each park. This was used in our milestone 2 project leading up to this for future park attendance prediction. It is unused in this project, but we've left the file in because the data shows up in some of our datasets, and this demonstrates where it came from.
- `NPS_weather_data_join.ipynb`
Holds all the dictionaries of all the weather stations that were queried for each year of temperature and precipitation data, and allows a user to see specifically which stations are used for the data for each park by year if the user is struggling with insomnia. All that data is then queried and written to csv files at the end.
- `NPS_weather_debugging.ipynb`
Shows the process for determining which weather stations have complete or incomplete data for a given year and data type, to then facilitate looking up the next nearest weather station on the [NOAA](https://www.ncdc.noaa.gov/cdo-web/search;jsessionid=7A87B303411A4E79CD8192D47B05F44D) website and building a complete dictionary to be queried in the weather_data_join notebook.
### Base directory files:
- `activities_scraper.ipynb`
Scrapes the listed activities for each park off of the [NPS](nps.gov) website and saves them to csv to then be used by the recommender engine to match park activities to the user's preferences.
- `analysis.ipynb`
Contains analysis of correlation matrix, silhouette scores, optimal number of clusters, and weather variance.
- `attendance_filter.ipynb`
Debugging notebook for filtering out parks based on their attendance numbers and the inputted user's preference.
- `capstone_SL.py`
This runs the Streamlit app. Contains the full end-to-end process of using all the various datasets, and establishing the UI to take the user input and present the results in an intuitive way. This is the combination of the work from many of the other notebooks. As the only `.py` it also serves to prove that we're capable of writing scripts, despite our campaign to convert the whole world to ipython notebooks.
- `park_names_cleaning.ipynb`
Takes care of a few cleaning operations, most specifically combining King's Canyon and Sequoia National Park data, which is discussed further in our blog post.
- `rec_sys.ipynb`
The pipeline of everything the recommender system does. From initial clustering, to filtering based on user criteria, to even evaluating  results to help us improve it.
- `README.md`
Forces the user to put up with my dull sense of humor.


To run the Streamlit app locally, you can use the command
```
streamlit run [path to the cloned repo]/capstone_SL.py
```
However, you will not get the lower dataframe which includes drive times, as you won't have access to the `secrets.toml` which contains our google maps api key...not that we don't trust you.

## Conclusion:
We made this project out of a love for the outdoors and national parks. We hope it showcases some cool things that we've been working on and some cool ways to explore the uniqueness of each national park. Most of all though, we hope it inspires you to get out and explore the national parks more. We hope that it can give you ideas of new places to visit and explore that you might not have previously considered. And in some small measure, we hope that this project can bring a little more joy and a little more wonder to your life. Now get out there and enjoy our wonderful parks, because as the late John Muir said:

> In every walk with nature one receives far more than he seeks.