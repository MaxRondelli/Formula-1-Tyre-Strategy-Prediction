import fastf1 as ff1
import pandas as pd
import numpy as np
import dict_data

#------------------------------------- First experiment -------------------------------------
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
        
        # count = 0 
        # for i, race in enumerate(race_list):
        #     if race == 'Spielberg':
        #         count_str = str(count)
        #         new_race = str(race+count_str)
    
        #         race_list.insert(i, new_race)
        #         race_list.remove(race)
        #         count += 1             
    elif year == 2020:
        race_list.remove('Montmeló')    
        race_list.remove('Montmeló')    

        # count = 0 
        # for i, race in enumerate(race_list):
        #     if race == 'Spielberg':
        #         count_str = str(count)
        #         new_race = str(race+count_str)
    
        #         race_list.insert(i, new_race)
        #         race_list.remove(race)
        #         count += 1
        #     elif race == 'Silverstone':
        #         count_str = str(count)
        #         new_race = str(race+count_str)
    
        #         race_list.insert(i, new_race)
        #         race_list.remove(race)
        #         count += 1
        #     elif race == 'Sakhir':
        #         count_str = str(count)
        #         new_race = str(race+count_str)
    
        #         race_list.insert(i, new_race)
        #         race_list.remove(race)
        #         count += 1

    return race_list
        
def load_dataset(year_list):
    driver_race_data_list = []
    driver_encoding = {}
    race_encoding = {}
    compound_encoding = {}

    for year in year_list:
        # Get the race list for the input year
        race_list = ['Imola'] #get_race_list(year) 

        driver_race_data = {}

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
            driver_race_data[key] = np.nan_to_num(value, nan=-1)

        driver_race_data_list.append(driver_race_data)

    return driver_race_data_list

def generate_dataset(year_list):

    driver_race_data_list = load_dataset(year_list)

    # Determine the shape of the 3D numpy array
    m, n = next(iter(driver_race_data_list[0].values())).shape
    N = sum(len(d) for d in driver_race_data_list)
    full_dataset = np.zeros((N, m, n))

    # Convert each dictionary to a 3D numpy array and stack them
    i = 0
    for dataset in driver_race_data_list:
        for key, value in dataset.items():
            full_dataset[i] = value
            i += 1

    # Save the full dataset to a file
    np.save('ex1_data.npy', full_dataset)

    return full_dataset

#------------------------------------- Second Experiment -------------------------------------

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
    if not lap_data_dict:
        return None
    
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
    
    best_compound = None 
    
    # It gets the compound at t time. 
    for i, entry in enumerate(compound):
        if i == t:
            best_compound = entry
            
    #print = f"Driver: {driver} - Compound: {best_compound}"
    return best_compound

# Dataset for final experiment: best tyre prediction
# Get information for a specific race and year. 
def get_information(session, race, year):
    # Get lap number for the race
    lap = dict_data.laps[race]
    
    # Weather conditions data
    air_temperature = session.laps.get_weather_data()['AirTemp']
    humidity = session.laps.get_weather_data()['Humidity']
    pressure = session.laps.get_weather_data()['Pressure']
    rainfall = session.laps.get_weather_data()['Rainfall']
    rainfall = np.where(rainfall == True, 1, 0)

    track_temperature = session.laps.get_weather_data()['TrackTemp']
    wind_direction = session.laps.get_weather_data()['WindDirection']
    wind_speed = session.laps.get_weather_data()['WindSpeed']
    
    year_list = [year] * lap
    race = [session.event['Location']] * lap   
    
    lap_list = []
    for i in range(lap):
        lap_list.append(i)
    
    list_of_tuples = list(zip(race, year_list, lap_list, air_temperature, humidity, pressure, rainfall, track_temperature, wind_direction, wind_speed))
    
    df = pd.DataFrame(list_of_tuples, columns = ['Race', 'Year', 'Lap', 'Air Temperature', 'Humidity', 
                                                 'Pressure', 'Rainfall', 'Track Temperature', 'Wind Direction', 
                                                 'Wind Speed'])
    
    return df 

def populate_dataset(year_list):
    np.set_printoptions(formatter={'float': lambda x: format(x, '.1f')}) # Output format

    dataset = []
    race_encoding = {}
    compound_encoding = dict_data.compound

    for year in year_list:
        # Get the race list for the current year
        race_list = get_race_list(year)
        
        dataset_data = {}
        
        for race in race_list:
            session = ff1.get_session(year, race, 'R')
            session.load()     
            
            # Get driver's information for the current session
            driver_information = get_information(session, race, year)
            
            # Encode and replace race data
            race_encoding[race] = dict_data.races[race]
            race_encoded = race_encoding[race]
            driver_information['Race'] = driver_information['Race'].replace(race, race_encoded)
            
            # Initialize lap data array for current race and year
            lap_data_array = np.full((len(driver_information['Lap']), 11), -1, dtype=np.float32)

            for i, lap in enumerate(driver_information['Lap']):
                lap_data = list(driver_information.loc[driver_information['Lap'] == lap].values[0])
                target = get_compound_for_time(session, lap) # Best compound at each lap
                
                if target is not None:
                    compound_encoded = compound_encoding[target] # Encoding the compound from string to integer
                elif target is np.NaN:
                    compound_encoded = -1
                else: 
                    compound_encoded = -1 
                    
                lap_data.append(compound_encoded)
                lap_data_array[i,:] = lap_data

            dataset_data[(race_encoded, year)] = lap_data_array
            
            while dataset_data[(race_encoded, year)].shape[0] < 78:
                lap = dataset_data[(race_encoded, year)].shape[0] + 1
                new_row = np.array([[race_encoded, year, lap, -1, -1, -1, -1, -1, -1, -1, -1]])
                dataset_data[(race_encoded, year)] = np.vstack((dataset_data[(race_encoded, year)], new_row))
    
        dataset.append(dataset_data)
        
    return dataset

# Returns a 3D numpy array
def get_dataset(year_list):
    
    dataset_dict = populate_dataset(year_list)
    
    # Determine the shape of the 3D numpy array
    m, n = next(iter(dataset_dict[0].values())).shape
    N = sum(len(d) for d in dataset_dict)
    full_dataset = np.zeros((N, m, n))

    # Convert each dictionary to a 3D numpy array and stack them
    i = 0
    for dataset in dataset_dict:
        for key, value in dataset.items():
            full_dataset[i] = value
            i += 1


    # Save the full dataset to a file
    np.save('exp2_final_data.npy', full_dataset)

    return full_dataset
