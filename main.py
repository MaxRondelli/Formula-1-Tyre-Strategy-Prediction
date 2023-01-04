import fastf1 as ff1
import pandas as pd
import numpy as np

# Enable the cache
ff1.Cache.enable_cache('Cache') # The argument is the name of the folder.

# Choose session and driver
session = ff1.get_session(2022, 'Imola', 'R') # Imola's race in 2022
driver = 'VER'  

session.load() # Load the session
 
drivers = pd.unique(session.laps['Driver']) # Array of all drivers
   
laps_driver = session.laps.pick_driver(driver) 

# Lap features
lap_time = laps_driver['LapTime'] # Session time when the lap was set (End of lap).
lap_numbers = laps_driver['LapNumber'] # That's our temporal resolution T. It's effective lenght of one unit of time. 

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

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df)

# Convert the encoded dataframe to a NumPy Array
array = df_encoded.to_numpy()

# Right now we have to understand how to build y. 
# y is a vector that contains [y_1, . . . , y_t]. At t time it has to contain the value of the best tyre for that time. 
# We gotta find out the metric system. The metric system is a characteristics that make a specific tyre better than another. 
  
# def choose_driver_with_best_lap_time(t):
#   # Load data on driver lap times in various racing conditions
#   lap_time_data = load_lap_time_data()

#   # Select data on driver lap times at time t
#   data_t = select_data_for_time(lap_time_data, t)

#   # Choose the driver with the best lap time among those available at time t
#   best_driver = choose_driver(data_t)

#   return best_driver

# def select_data_for_time(data, t):
#   # Initialize an empty list to store the selected data
#   data_t = []

#   # Iterate over each entry in the data
#   for entry in data:
#     # If this entry is for the desired time t, add it to the list
#     if entry["time"] == t:
#       data_t.append(entry)

#   return data_t

# def choose_driver(data):
#   # Initialize the best driver with the first driver in the data
#   best_driver = data[0]["driver"]
#   best_lap_time = data[0]["lap time"]

#   # Iterate over each driver in the data
#   for driver in data:
#     # If this driver's lap time is better than the best lap time of the current best driver,
#     # update the best driver
#     if driver["lap time"] < best_lap_time:
#       best_driver = driver["driver"]
#       best_lap_time = driver["lap time"]

#   return best_driver



# # Determine the metric to use for choosing the best driver
# metric = "lap time"

# y = [] # Initialize the y vector

# for t in range(lap_numbers):
#     # Choose the valuo of y_t based on the matric
#     if metric == "lap time":
#         y_t = choose_driver_with_best_lap_time(t)
    
#     # Add y_t to the y vector
#     y.append(y_t)
    
    
    
def load_lap_time_data():
    # Initialize an empty dictionary to store lap times for each driver
    lap_times = {}

    for driver in drivers:
        lap_times[driver] = []
        for lap in lap_numbers:
            d = session.laps.pick_driver(driver)
            lap_time = d['LapTime']
            
            # Add the lap time to the list for this driver
            lap_times[driver].append(lap_time)
                       
    return lap_times

print(load_lap_time_data())
