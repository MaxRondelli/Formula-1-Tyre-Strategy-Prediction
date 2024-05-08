import fastf1 as ff1    
import pandas as pd
import warnings 
import math
import csv 
# Enable fastf1 cache
ff1.Cache.enable_cache('cache')

""" 
Getting information about weather and track condition are important to predict which tyre is the best at each lap. 
Recurrent Neural Networks (RNN) are used for this experiment. Times series are the most useful net's for this task. 

LAP | COMPOUND | WEATHER INFORMATIONS

"""

session = ff1.get_session(2023, 'Sakhir', 'R')
session.load()

print(session.laps.pick_lap(3).pick_fastest()['TyreLife'])

total_laps = session.total_laps # Get total laps for the session

# Lists to store data for DataFrame
lap_list = []
compound_list = []
lap_time_list = []
lap_data = []

"""
The value could miss and being nan. It will be replaced as an empty string.
Return: compound and weather data for a specific fastest lap if available. Otherwise, empty string. 
"""
def fastestDriverData(lap):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="In the future, `None` will be returned instead of an empty `Lap` object")
        compound = session.laps.pick_lap(lap).pick_fastest()['Compound']

        if not isinstance(compound, float) or not math.isnan(compound):
            weather_data = session.laps.pick_lap(lap).pick_fastest().get_weather_data()
            weather_data = weather_data[['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']]
            # Format Time 
            lap_time = session.laps.pick_lap(lap).pick_fastest()['LapTime']
            minutes = lap_time.seconds // 60
            seconds = lap_time.seconds % 60
            milliseconds = lap_time.microseconds // 1000
            lap_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

            return compound, weather_data, lap_time
        else:
            return ""

def createDataframe(total_laps):
    # Create an empty list to store lap data
    lap_data = []
    
    # Iterate over laps
    for lap in range(total_laps):
        # Get data using fastestDriverData function
        data = fastestDriverData(lap)
        
        # If lap data is available
        if data:
            compound, weather_data, lap_time = data
            
            # Construct lap dictionary
            lap_dict = {
                'LAP': lap,
                'COMPOUND': compound,
                'AIR TEMP': weather_data.get('AirTemp', ""),
                'HUMIDITY': weather_data.get('Humidity', ""),
                'PRESSURE': weather_data.get('Pressure', ""),
                'RAINFALL': weather_data.get('Rainfall', ""),
                'TRACK TEMP': weather_data.get('TrackTemp', ""),
                'WIND DIRECTION': weather_data.get('WindDirection', ""),
                'WIND SPEED': weather_data.get('WindSpeed', ""),
                'LAP TIME': lap_time
            }
            
            # Append lap data to the list
            lap_data.append(lap_dict)
    
    # Create DataFrame
    df = pd.DataFrame(lap_data)
    
    # Save DataFrame to a text file (CSV format)
    df.to_csv('lap_data.txt', sep='\t', index=False, quoting=csv.QUOTE_NONE)

createDataframe(total_laps)