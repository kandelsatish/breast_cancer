"""
Global variables
"""



import json
import os

global app_name
global log_level



# get the path of the config file
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file_path = os.path.join(root_dir, "config", "config.json")


# load config file
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


config_file=load_config(config_file_path)

app_name=config_file['__name__']
log_level="logging."+config_file['log_level']
