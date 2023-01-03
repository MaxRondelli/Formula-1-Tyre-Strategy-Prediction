import fastf1 as ff1
import pandas as pd
import numpy as np

# Enable the cache
ff1.Cache.enable_cache('Cache') # The argument is the name of the folder.

# Choose session and driver
session = ff1.get_session(2022, 'Imola', 'R') # Imola's race in 2022
driver = 'VER'  

session.load() # Load the session

laps_driver = session.laps.pick_driver(driver) 

# Lap features
lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap).
lap_numbers = laps_driver['LapNumber'] #  That's our temporal resolution T. It's effective lenght of one unit of time. 

# Tyre features
tyre_life = laps_driver['TyreLife'] # Laps driven on this tyre. It includes laps in other session for used sets of tyre.
compound = laps_driver['Compound'] # Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
stint = laps_driver['Stint'] # Stint number

# Weather conditions features
weather_rainfall = session.laps.get_weather_data()['Rainfall'] # Shows if there is rainfall
weather_track_temperature = session.laps.get_weather_data()['TrackTemp'] # Track temperature [°C]

# Create DataFrame
list_of_tuples = list(zip(lap_time, lap_numbers, tyre_life, compound, stint, weather_rainfall, weather_track_temperature))
df = pd.DataFrame(list_of_tuples, columns = ['Lap', 'Time', 'Compound', 'Tyre Life', 'Stint', 'Rainfall', 'Track Temp'])
print(df.to_markdown())