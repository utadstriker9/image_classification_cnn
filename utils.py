import yaml
import pandas as pd
from datetime import datetime

CONFIG_DIR = 'config.yaml'

# Function to load config files
def config_load():
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameter file not found in path')
    return config

# Function to load json files
def load_json(file_path):
    data = open(file_path)
    return pd.read_json(data)

# Function to dump data into json
def dump_json(data, file_path):
    data.to_json(file_path)
    
# Function that return the current time
def time_stamp():
    return datetime.now()