import fastf1 as ff1
import pandas as pd
import numpy as np
import dict_data

def timedelta_to_seconds(td):
    return td / np.timedelta64(1, 's')

# Function creates a df for a specific driver and session. 
def get_data(driver, session):
    session_driver = session.laps.pick_driver(driver)
    
    driver_lap_number = session_driver['LapNumber'] # Driver's lap  
    driver_sector1_time = (session_driver['Sector1Time'] / np.timedelta64(1, 's')).astype(float) # Sector 1 recorded time
    driver_sector2_time = (session_driver['Sector2Time'] / np.timedelta64(1, 's')).astype(float) # Sector 2 recorded time
    driver_sector3_time = (session_driver['Sector3Time'] / np.timedelta64(1, 's')).astype(float) # Sector 3 recorded time
    driver_lap_time = session_driver['LapTime'].apply(timedelta_to_seconds)
    
    weather_rainfall = session.laps.get_weather_data()['Rainfall'] # Shows if there is rainfall
    weather_rainfall = np.where(weather_rainfall == True, 1, 0)
    weather_track_temperature = session.laps.get_weather_data()['TrackTemp'] # Track temperature [°C]
       
    driver_list = [driver] * len(driver_lap_number)
    grand_prix_list = [session.event['Location']] * len(driver_lap_number)   
    
    compound = session_driver['Compound']
    
    list_of_tuples = list(zip(driver_list, grand_prix_list, driver_lap_number, driver_sector1_time, driver_sector2_time, driver_sector3_time, driver_lap_time, weather_rainfall, weather_track_temperature, compound))
    df = pd.DataFrame(list_of_tuples, columns = ['Driver', 'Race', 'Lap', 'Sector 1 Time', 'Sector 2 Time', 'Sector 3 Time', 'Lap Time', 'Rainfall', 'Track Temp', 'Compound']) 
    
    return df     

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
        race_list.remove('Sakhir')  
        
    return race_list
        
def load_dataset(year_list):
    driver_race_data = {}
    driver_encoding = {}
    race_encoding = {}
    compound_encoding = {}

    for year in year_list:
        # Get the race list for the input year
        race_list = get_race_list(year) 
             
        for race in race_list:
            session = ff1.get_session(year, race, 'R')
            session.load()
            driver_list = pd.unique(session.laps['Driver'])

            for driver in driver_list:
                session_driver = session.laps.pick_driver(driver)
                
                # Load all the driver's information for the current session
                data = get_data(driver, session)

                # Encode and replace driver data.
                driver_encoding[driver] = dict_data.drivers[driver]
                driver_encoded = driver_encoding[driver]
                data['Driver'] = data['Driver'].replace(driver, driver_encoded)
                
                # Encode and replace race data.
                race_encoding[race] = dict_data.races[race]
                race_encoded = race_encoding[race]
                data['Race'] = data['Race'].replace(race, race_encoded)
                
                # Compound's driver data from fastf1 library. 
                compound_list = session_driver['Compound']
                
                for compound in compound_list:
                    
                    # Encode and replace compound data.
                    compound_encoding[compound] = dict_data.compound.get(compound, -1)
                    compound_encoded = compound_encoding[compound]
                    data['Compound'] = data['Compound'].replace(compound, compound_encoded) 
                
                    driver_race_data[(driver_encoded, race_encoded)] = data.values   
                    
                    # Add rows until lap is equal to 78 (Monaco's grand prix lap). 
                    while(driver_race_data[(driver_encoded, race_encoded)].shape[0] < 78):
                        lap = driver_race_data[(driver_encoded, race_encoded)].shape[0] + 1
                        new_row = np.array([[driver_encoded, race_encoded, lap, -1, -1, -1, -1, -1, -1, -1]])
                        driver_race_data[(driver_encoded, race_encoded)] = np.vstack(
                            (driver_race_data[(driver_encoded, race_encoded)], new_row))
        
        # Replace NaN values with -1
        for key, value in driver_race_data.items():
            driver_race_data[key] = np.nan_to_num(value, nan = -1)

    return driver_race_data

def generate_dataset(year_list):

    dataset = load_dataset(year_list)
    
    # Replace NaN values with -1
    for key, value in dataset.items():
        dataset[key] = np.nan_to_num(value, nan = -1)

    # Create 3D numpy array. Concatenate just the dataset's values 
    tmp_array = next(iter(dataset.values()))
    m, n = tmp_array.shape
    N = len(dataset)
    data = np.zeros((N, m, n)) # Initialize final dataset

    for i, key in enumerate(dataset.keys()):
        data[i, :, :] = dataset[key]     
    
    np.save('data2.npy', data) 
    return data


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
            
    print = f"Driver: {driver} - Compound: {best_compound}"
    return best_compound

'''
Creare nuovo dataset che fissata la gare e un anno ha per colonne:
    input: gli ripetiamo la gara e anno, lap, tutte le informazioni che non dipendono dal pilota (weather conditions etc.), life time del compound 
    output: ultima colonna, il compound migliore per quel giro e per quella gara
    
    (Imola, 2022) = [[Imola, 2022, 1, .........., 'soft']]
'''
# Dataset for final experiment: best tyre prediction
def dataframe(session, driver):
    session_driver = session.laps.pick_driver(driver)
    driver_lap_number = session_driver['LapNumber']
    
    # Weather conditions data
    air_temperature = session.laps.get_weather_data()['AirTemp']
    humidity = session.laps.get_weather_data()['Humidity']
    pressure = session.laps.get_weather_data()['Pressure']
    rainfall = session.laps.get_weather_data()['Rainfall']
    track_temperature = session.laps.get_weather_data()['TrackTemp']
    wind_direction = session.laps.get_weather_data()['WindDirection']
    wind_speed = session.laps.get_weather_data()['WindSpeed']
    
    race = [session.event['Location']] * len(driver_lap_number)   
    
    list_of_tuples = list(zip(race, driver_lap_number, air_temperature, humidity, pressure, rainfall, track_temperature, wind_direction, wind_speed))
    
    df = pd.DataFrame(list_of_tuples, columns = ['Race', 'Lap', 'Air Temperature', 'Humidity', 
                                                 'Pressure', 'Rainfall', 'Track Temperature', 'Wind Direction', 
                                                 'Wind Speed'])
    
    return df 
def dataset(race_list, year):
    race_data = {}
    
    for race in race_list:
        session = ff1.get_session(year, race, 'R')
        session.load()
        driver_list = pd.unique(session.laps['Driver'])

        for driver in driver_list:
            data = dataframe(session, driver)
            
            race_data[(race, year)] = data.values
        
    return race_data
        
               
        
