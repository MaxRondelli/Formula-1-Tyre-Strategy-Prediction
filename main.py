import fastf1 as ff1
import pandas as pd
import numpy as np
import dict_data
import pickle
from utils import *
from LSTM import *
import sys 
# session = ff1.get_session(2022, 'Imola', 'R')
# driver = 'VER'
# # total_lap = 63 # That's our temporal resolution T. It's effective lenght of one unit of time.
# session.load()

# # # Lap features
# laps_driver = session.laps.pick_driver(driver) 
# # lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap) for the specific driver. 
# lap_numbers = laps_driver['LapNumber'] # Driver's lap
# print(len(lap_numbers))
# # Tyre features
# tyre_life = laps_driver['TyreLife'] # Laps driven on this tyre. It includes laps in other session for used sets of tyre.
# compound = laps_driver['Compound'] # Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
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

ff1.Cache.enable_cache('Cache')

race_list = get_race_list()
np.set_printoptions(threshold=sys.maxsize)

x = generate_dataset(race_list)

# print(x.shape)
print(x)