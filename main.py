import fastf1 as ff1
import pandas as pd
import numpy as np
from utils import *

# Enable the cache
ff1.Cache.enable_cache('Cache') # The argument is the name of the folder.

session = ff1.get_session(2022, 'Austria', 'R')
driver = 'VER'
total_lap = 63 # That's our temporal resolution T. It's effective lenght of one unit of time.

session.load()

# Lap features
laps_driver = session.laps.pick_driver(driver) 
lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap) for the specific driver. 
lap_numbers = laps_driver['LapNumber'] # Driver's lap

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

print(df.to_markdown())

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df)

# Convert the encoded dataframe to a NumPy Array
array = df_encoded.to_numpy()
    
# print(get_compound_for_time(session, 21))

'''
lanciare nel main genearete_df sulla lista dei tracciati (tutti) e salvare output con la libreria pandas. (save.csv)
'''