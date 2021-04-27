#!/usr/bin/python3.7
from data_utils import *

path = 'Training_Data'
data = import_data_info(path=path, start_folder=0, end_folder=1)

visualize_balance_data(data, display=True, balance=True)
