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

# session = ff1.get_session(2021, 'Zandvoort', 'R')
# driver = 'VER'
# session.load()
# laps_driver = session.laps.pick_driver(driver) 
# compound = laps_driver['Compound'] # Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)

# df = pd.DataFrame(compound, columns = ['Compound'])
# print(df.to_markdown())

# with open('dataset.pickle', 'wb') as f:
#     pickle.dump(array, f)


year_list = [2021, 2022]
x = generate_dataset(year_list)
print(x)

# print(get_race_list(2021))
# grand_prix_list = ff1.get_event_schedule(2021)
# print(grand_prix_list['Location'])

# race_list = get_race_list(2021)
# dataset = dataset(race_list, 2021)
# print(dataset)