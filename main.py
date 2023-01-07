import fastf1 as ff1
import pandas as pd
import numpy as np

# Enable the cache
ff1.Cache.enable_cache('Cache') # The argument is the name of the folder.

def select_session(year, grand_prix, session_type):
    session = ff1.get_session(year, grand_prix, session_type) 
    return session

def select_driver(driver):
    return driver

session = select_session(2022, 'Austria', 'R')
driver = select_driver('VER')  

session.load() # Load the session
 
# Lap features
laps_driver = session.laps.pick_driver(driver) 
lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap) for the specific driver. 
lap_numbers = laps_driver['LapNumber'].to_numpy() # That's our temporal resolution T. It's effective lenght of one unit of time. 

# Tyre features
tyre_life = laps_driver['TyreLife'] # Laps driven on this tyre. It includes laps in other session for used sets of tyre.
compound = laps_driver['Compound'] # Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
stint = laps_driver['Stint'] # Stint number

# Weather conditions features
weather_rainfall = session.laps.get_weather_data()['Rainfall'] # Shows if there is rainfall
weather_track_temperature = session.laps.get_weather_data()['TrackTemp'] # Track temperature [°C]

# Create DataFrame
list_of_tuples = list(zip(lap_numbers, lap_time, compound, tyre_life , stint, weather_rainfall, weather_track_temperature))
df = pd.DataFrame(list_of_tuples, columns = ['Lap', 'Time', 'Compound', 'Tyre Life', 'Stint', 'Rainfall', 'Track Temp'])

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df)

# Convert the encoded dataframe to a NumPy Array
array = df_encoded.to_numpy()

# Function returns all lap times for each lap for each driver
def get_lap_times():
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
def get_best_driver_for_time(t):
    # Load data on driver lap times.
    lap_data = get_lap_times()
    
    # Select data on driver lap times at time t
    data_t = get_data_for_time(lap_data, t)
    
    # Choose the driver with the best lap time among those available at time t
    best_driver = get_best_driver(data_t)
    
    return best_driver

# Function returns the compound of the driver how had the best time at t time. 
def get_compound_for_time(t):   
    driver = get_best_driver_for_time(t)  
    
    session_driver = session.laps.pick_driver(driver)
    compound = session_driver['Compound']
    
    best_compound = 0 
    
    # It gets the compound at t time. 
    for i, entry in enumerate(compound):
        if i == t:
            best_compound = entry
            
    return best_compound

print(get_compound_for_time(21))