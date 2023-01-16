import fastf1 as ff1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function returns all lap times for each lap for each driver
def get_lap_times(session):
    lap_time_data_dict = [] # Create a list of dictionaries
    drivers = pd.unique(session.laps['Driver']) # Array of all drivers

    # Iterate over the tuples in the list
    for driver in drivers:
        # Get all lap time for the current driver
        session_driver = session.laps.pick_driver(driver)
        lap_time = session_driver['LapTime']
        count = 0
        for lap in lap_time:    
            # Create a new dictionary for each tuple
            lap_dict = {'driver': driver, 'lap': count ,'lap_time': lap}
            
            if pd.notnull(lap): # Value != NaT. The first lap is not in the dict. 
                lap_time_data_dict.append(lap_dict) # Add the dictionary to the list
            
            # Update count that it's the lap's number.
            count = count + 1 
                    
    return lap_time_data_dict

# Function selects data on driver lap times at time t from the full lap time data. It returns information for all driver at the specific lap t. 
def get_data_for_time(lap_data_dict, t):
    data_t = [] # Empty list to store the selected data
    
    for entry in lap_data_dict:
        if entry["lap"] == t:
            data_t.append(entry)
            
    return data_t

# Function returns all data for one specific given driver.
def get_driver_data(lap_data_dict, driver):  
    get_driver_data = []
     
    for entry in lap_data_dict:
        if(driver == entry['driver']):
            driver = entry['driver']
    
    for entry in lap_data_dict:
        if entry.get('driver') == driver:
            get_driver_data.append(entry)
    
    return get_driver_data

# Function returns all data for one specific driver at one time t.
def get_driver_data_for_time(lap_data_dict, driver, t):
    driver_data_t = []  
    
    for entry in lap_data_dict:
        if entry['driver'] == driver:
            if entry['lap'] == t:
                driver_data_t.append(entry)
    
    return driver_data_t    

# Function choose driver with the best lap time among those available at t time, using the lap time data provided as input.
def get_best_driver(lap_data_dict):
    best_driver = lap_data_dict[0]['driver']
    best_lap_time = lap_data_dict[0]['lap_time']

    for entry in lap_data_dict:
        # If this driver's lap time is better than the best lap time of the current best driver, update the best driver.
        if entry['lap_time'] < best_lap_time:
            best_lap_time = entry['lap_time']
            best_driver = entry['driver']
            
    return best_driver

# Function returns the driver with the best lap time at time t, based on available data on driver lap times.  
def get_best_driver_for_time(session, t):
    # Load data on driver lap times.
    lap_data = get_lap_times(session)
    
    # Select data on driver lap times at time t
    data_t = get_data_for_time(lap_data, t)
    
    # Choose the driver with the best lap time among those available at time t
    best_driver = get_best_driver(data_t)
    
    return best_driver

# Function returns the compound of the driver how had the best time at t time. 
def get_compound_for_time(session, t):   
    driver = get_best_driver_for_time(session, t)  
    
    session_driver = session.laps.pick_driver(driver)
    compound = session_driver['Compound']
    
    best_compound = 0 
    
    # It gets the compound at t time. 
    for i, entry in enumerate(compound):
        if i == t:
            best_compound = entry
            
    return f"Driver: {driver} - Compound: {best_compound}"


def get_data(driver, session):
    session_driver = session.laps.pick_driver(driver)

    driver_lap_number = session_driver['LapNumber'] # Driver's lap  

    driver_sector1_time = (session_driver['Sector1Time'] / np.timedelta64(1, 's')).astype(float) # Sector 1 recorded time
    driver_sector2_time = (session_driver['Sector2Time'] / np.timedelta64(1, 's')).astype(float) # Sector 2 recorded time
    driver_sector3_time = (session_driver['Sector3Time'] / np.timedelta64(1, 's')).astype(float) # Sector 3 recorded time
    driver_lap_time = (session_driver['LapTime'] / np.timedelta64(1, 's')).astype(float) # Lap Time recorded time
    
    weather_rainfall = session.laps.get_weather_data()['Rainfall'] # Shows if there is rainfall
    weather_track_temperature = session.laps.get_weather_data()['TrackTemp'] # Track temperature [°C]
    
    list_of_tuples = list(zip(driver_lap_number, driver_sector1_time, driver_sector2_time, driver_sector3_time, driver_lap_time, weather_rainfall, weather_track_temperature))
    df = pd.DataFrame(list_of_tuples, columns = ['Lap', 'Sector 1 Time', 'Sector 2 Time', 'Sector 3 Time', 'Lap Time', 'Rainfall', 'Track Temp'])
    
    return df 
    

def generate_df(race_list):
    '''
    race_list è una lista con i nomi di tutta le gare --> race_list = ['Imola', 'Monza', ...]
    final_df = pd.DataFrame(columns = ['Driver, Race, le stesse colonne di get_data()'])
    
    for race_name in race_list:
        session = ff1.get_session(2022, race_name, 'R')
        driver_list = lista dei driver che hanno participato alla session. 
        
        for driver in driver_list: 
            data = get_data(driver, session)
            
            A data bisogna aggiungere due colonne all'inizio chiamate driver e race contenenti tante copie quante
            righe ha data tutte uguali con valore il nome del driver e race attuale.
            
            - creare un df nuovo con lo stesso numero di righe di data e due colonne, driver e race a valore costante. 
            - data = pd.Concatenazione_Orizzonatale(df, data)
            
            sempre dentro il ciclo di driver, bisogna concatenare in verticale il df data appena aggiornato 
            con il dataframe finale che viene inizializzato sopra. 
            
            final_df = pd.Concatenazione_Verticale(final_df, data)         
    return final_df       
    '''
    
    final_df = pd.DataFrame()    
    for race_name in race_list:
        session = ff1.get_session(2022, race_name, 'R')
        session.load()
        driver_list = pd.unique(session.laps['Driver']) # Array of all drivers

        for driver in driver_list:
            data = get_data(driver, session)

            tmp_df = pd.DataFrame(index = data.index, columns = ['Driver', 'Race'])
            tmp_df['Driver'] = driver 
            tmp_df['Race'] = race_name   
            
            data = pd.concat([data, tmp_df], axis = 1)         
            
            final_df = pd.concat([final_df, data], ignore_index=True)
                
    columns = final_df.columns.tolist()
    columns.remove('Driver')
    columns.remove('Race')
    
    final_df = final_df[['Driver', 'Race'] + columns]
    return final_df