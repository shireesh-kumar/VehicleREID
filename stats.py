import json
from collections import defaultdict

# Load the JSON data from a file
with open('/home/sporalas/VehicleREID/test_data.json', 'r') as file: 
    data = json.load(file)

# Initialize counters for color and type
color_count = defaultdict(int)
type_count = defaultdict(int)

# Iterate over the data to count colors and types
for entry in data.values():
    color_id = entry['colorID']
    type_id = entry['typeID']
    
    color_count[color_id] += 1
    type_count[type_id] += 1

# Print the results
print("Total counts per color:")
for color, count in color_count.items():
    print(f"Color ID {color}: {count} vehicles")

print("\nTotal counts per type:")
for type_id, count in type_count.items():
    print(f"Type ID {type_id}: {count} vehicles")
