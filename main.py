import fastf1 as ff1
import pandas as pd
import numpy as np
import dict_data
import pickle
from utils import *
from LSTM import *
import sys 

# Enable fastf1 cache
ff1.Cache.enable_cache('Cache')

# session = ff1.get_session(2022, 'Imola', 'R')
# driver = 'VER'
# # # total_lap = 63 # That's our temporal resolution T. It's effective lenght of one unit of time.
# session.load()

# # # # Lap features
# laps_driver = session.laps.pick_driver(driver) 
# # # lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap) for the specific driver. 
# # lap_numbers = laps_driver['LapNumber'] # Driver's lap
# # print(len(lap_numbers))
# # # Tyre features
# # tyre_life = laps_driver['TyreLife'] # Laps driven on this tyre. It includes laps in other session for used sets of tyre.
# compound = laps_driver['Compound'] # Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
# print(compound)
# stint = laps_driver['Stint'] # Stint number
# # Weather conditions features
# weather_rainfall = session.laps.get_weather_data()['Rainfall'] # Shows if there is rainfall
# weather_track_temperature = session.laps.get_weather_data()['TrackTemp'] # Track temperature [°C]
# # Create DataFrame
# list_of_tuples = list(zip(lap_numbers, lap_time, compound, tyre_life , stint, weather_rainfall, weather_track_temperature))
# df = pd.DataFrame(list_of_tuples, columns = ['Lap', 'Time', 'Compound', 'Tyre Life', 'Stint', 'Rainfall', 'Track Temp'])
# #print(df.to_markdown())

# with open('dataset.pickle', 'wb') as f:
#     pickle.dump(array, f)

def get_numpy_dataset():
    np.set_printoptions(threshold=sys.maxsize)   
    
    year_list = [2021, 2022]
    for i in year_list:  
        race_list = get_race_list(i)
        x = generate_dataset(race_list, i)

    return x

get_numpy_dataset()
print(get_numpy_dataset())
# grand_prix_list = ff1.get_event_schedule(2021)
# print(grand_prix_list, race_list)

# race_list = get_race_list(2021)
# dataset = dataset(race_list, 2021)
# print(dataset)