import fastf1 as ff1    
import pandas as pd
import warnings 
import math
import csv 
import os
import numpy as np

# Enable fastf1 cache
ff1.Cache.enable_cache('cache')

# Lists to store data for DataFrame
def fastestDriverData(lap):
    COMPOUNDS = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4} # Compounds dictionary. It is necessary to convert compound string to int
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="In the future, `None` will be returned instead of an empty `Lap` object")
        compound = session.laps.pick_lap(lap).pick_fastest()['Compound']

        if not isinstance(compound, float) or not math.isnan(compound):
            # Enconding the compound from string to int
            if compound is not None:
                compound = COMPOUNDS[compound]
            elif compound is np.NaN or compound is None:
                compound = -1 
            if compound == -1:
                print(f"Warning: Compound '{compound}' not found in COMPOUNDS dictionary")
            
            # Weather data
            weather_data = session.laps.pick_lap(lap).pick_fastest().get_weather_data()
            weather_data = weather_data[['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']]

            if weather_data['Rainfall'] == True:
                weather_data['Rainfall'] = 1
            else: 
                weather_data['Rainfall'] = 0

            # Format Time 
            lap_time = session.laps.pick_lap(lap).pick_fastest()['LapTime']
            lap_time = pd.to_timedelta(lap_time).total_seconds() # convert lap time from api format to seconds

            return compound, weather_data, lap_time

def createDataframe(total_laps):
    # Create an empty list to store data for DataFrame
    lap_data = []

    # Iterate over laps
    for lap in range(total_laps):
        # Get data using fastestDriverData function
        data = fastestDriverData(lap)
        
        # If lap data is available
        if data:
            compound, weather_data, lap_time = data
            
            # Construct lap dictionary
            lap_dict = {
                'LAP': lap,
                'AIR TEMP': weather_data.get('AirTemp', ""),
                'HUMIDITY': weather_data.get('Humidity', ""),
                'PRESSURE': weather_data.get('Pressure', ""),
                'RAINFALL': weather_data.get('Rainfall', ""),
                'TRACK TEMP': weather_data.get('TrackTemp', ""),
                'WIND DIRECTION': weather_data.get('WindDirection', ""),
                'WIND SPEED': weather_data.get('WindSpeed', ""),
                'LAP TIME': lap_time,
                'COMPOUND': compound
            }

            # Append lap data to the list
            lap_data.append(lap_dict)
    
    # Create DataFrame
    df = pd.DataFrame(lap_data)

    # Write on txt file
    file_exists = os.path.isfile('lap_data.txt')

    # If the file exists, read the existing data
    if file_exists:
        existing_df = pd.read_csv('lap_data.txt', sep='\t', quoting=csv.QUOTE_NONE)
        # Append the new data to the existing data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df
    combined_df.to_csv('lap_data.txt', sep='\t', index=False, quoting=csv.QUOTE_NONE)

    # Numpy file 
    npy_file_exists = os.path.isfile('lap_data.npy') # Check if the .npy file exists

    # If the .npy file exists, load the existing data
    if npy_file_exists:
        existing_data = np.load('lap_data.npy', allow_pickle=True)
        # Convert the existing data to DataFrame
        existing_df = pd.DataFrame(existing_data, columns=combined_df.columns)
        # Append the new data to the existing data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df   

    # Convert the combined data to numpy array
    data_array = combined_df.values

    # Save the data as .npy file
    np.save('lap_data.npy', data_array)

def get_race_list(year):
    grand_prix_list = ff1.get_event_schedule(year)
    race_list = []
                     
    for race in grand_prix_list['Location']:
        race_list.append(race)  
            
    # Removing Pre-season test sessions.
    if year == 2022:
        race_list.remove('Spain')
        race_list.remove('Bahrain')
    elif year == 2021:
        race_list.remove('Bahrain') 
    elif year == 2020:
        race_list.remove('Barcelona')    
        race_list.remove('Barcelona')   
    elif year == 2023:
        race_list.remove('Sakhir') 
    return race_list

years = [2020, 2021, 2022, 2023]
for year in years:
    race_list = get_race_list(year)
    for race in race_list:
        session = ff1.get_session(year, race, 'R')
        session.load()
        total_laps = session.total_laps # Get total laps for the session
        createDataframe(total_laps)