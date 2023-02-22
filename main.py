import fastf1 as ff1
import pandas as pd
import numpy as np
from utils import *
from LSTM import *

# Enable fastf1 cache
ff1.Cache.enable_cache('Cache')

#--------------- Generate .npy file for the first experiment ---------------
# year_list = [2022]
# x = load_dataset(year_list)
# print(x)

#--------------- Generate .npy file for the second experiment ---------------
# session = ff1.get_session(2019, 'Imola', 'R')
# session.load()
# driver = 'HAM'

# driver_session = session.laps.pick_driver(driver)
# compound = driver_session['Compound']
# print(compound)
# driver_list = pd.unique(session.laps['Driver'])
# print(driver_list)

# year_list = [2019, 2020, 2021, 2022]
# # for year in year_list:
# #     df = get_information(session, driver, year)
# #     print(df.to_markdown())
    
# x = get_dataset(year_list)
# print(x)

# # grand_prix_list = ff1.get_event_schedule(2020)
# # print(grand_prix_list)

# print(get_race_list(2020))