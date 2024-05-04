import fastf1 as ff1    

# Enable fastf1 cache
ff1.Cache.enable_cache('cache')

""" 
Getting information about weather and track condition are important to predict which tyre is the best at each lap. 
Recurrent Neural Networks (RNN) are used for this experiment. Times series are the most useful net's for this task. 
"""

session = ff1.get_session(2023, 'Sakhir', 'R')
session.load()

print(session.lap_count)

# # Create the dataset from API 
# YEARS = [2023]
# for year in YEARS:
#     race_list = ff1.get_event_schedule(year) # Get all the race for a specific year.
#     race_list = race_list["Location"][1:] # Not getting first location since it's test race.

#     for race in race_list:
#         session = ff1.get_session(year, race, 'R') # 'R' stands for race. We get only info of race session.
#         session.load()
#         total_laps = session.total_laps # Get total laps for that specific race.
        
#         for lap in range(total_laps):
#             # Get driver's information for the current session
#             air_temperature = session.laps.get_weather_data()['AirTemp']
#             print(air_temperature)

