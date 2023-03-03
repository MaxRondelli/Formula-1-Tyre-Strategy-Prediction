import fastf1 as ff1
from Utils import *

# Enable fastf1 cache
ff1.Cache.enable_cache('Cache')

#--------------- Generate .npy file for the first experiment ---------------
year_list = [2021, 2022]
exp1_data = load_dataset(year_list)

#--------------- Generate .npy file for the second experiment ---------------
year_list = [2019, 2020, 2021, 2022]
exp2_final_data = get_dataset(year_list)
