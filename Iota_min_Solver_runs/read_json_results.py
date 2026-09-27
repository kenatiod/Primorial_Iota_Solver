# read_json_results.py
"""
This program reads the json file from iota_min_results.py

By Ken Clements, Sept. 8, 2026
"""
import json

RESULTS_FILE = 'iota_min_results.json'

# Open the JSON file and load its contents
with open(RESULTS_FILE, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Now 'data' is a standard Python dictionary or list
#for key in data:
#    print({key})

iota_min_list = [res['iota_min']  for res in data['results'] if res['iota_min'] != None]
print(iota_min_list)
print(None)


