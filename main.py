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

# Create DataFrame
list_of_tuples = list(zip(lap_time, lap_numbers, tyre_life, compound, stint, weather_rainfall, weather_track_temperature))
df = pd.DataFrame(list_of_tuples, columns = ['Lap', 'Time', 'Compound', 'Tyre Life', 'Stint', 'Rainfall', 'Track Temp'])

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df)

# Convert the encoded dataframe to a NumPy Array
array = df_encoded.to_numpy()

# ------ Create y output vector ------
metric = lap_numbers # Determine the metric to use for choosing the best driver
y = [] # Output vector 

# Function choose driver with the best lap time among those available at t time, using the lap time data provided as input.
def choose_driver(lap_data_dict):
    best_driver = lap_data_dict[0]['driver']
    best_lap_time = lap_data_dict[0]['lap_time']

    for driver in lap_data_dict:
        # If this driver's lap time is better than the best lap time of the current best driver, update the best driver.
        if driver['lap_time'] < best_lap_time:
            best_lap_time = driver['lap_time']
            best_driver = driver['driver']
            
    return best_driver, best_lap_time

# Function returns all data for one specific given driver.
def get_driver_data(lap_data_dict, driver):  
    get_driver_data = []
     
    for driver_data in lap_data_dict:
        if(driver == driver_data['driver']):
            driver = driver_data['driver']
    
    for driver_data in lap_data_dict:
        if driver_data.get('driver') == driver:
            get_driver_data.append(driver_data)
    
    return get_driver_data
                
lap_data = get_lap_times() # Returns all lap time for each driver. It's a dictionary. 
best_lap_time = choose_driver(lap_data) # Returns the best's lap of the grand prix and the driver. Purple lap. 
get_driver_data(lap_data, 'LEC')
