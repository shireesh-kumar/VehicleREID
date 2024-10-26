import json
from collections import defaultdict

# # Load the JSON data from a file
# with open('/home/sporalas/VehicleREID/test_data.json', 'r') as file: 
#     data = json.load(file)

# # Initialize counters for color and type
# color_count = defaultdict(int)
# type_count = defaultdict(int)

# # Iterate over the data to count colors and types
# for entry in data.values():
#     color_id = entry['colorID']
#     type_id = entry['typeID']
    
#     color_count[color_id] += 1
#     type_count[type_id] += 1

# # Print the results
# print("Total counts per color:")
# for color, count in color_count.items():
#     print(f"Color ID {color}: {count} vehicles")

# print("\nTotal counts per type:")
# for type_id, count in type_count.items():
#     print(f"Type ID {type_id}: {count} vehicles")



# Load the JSON data from a file
with open('/home/sporalas/VehicleREID/test_data.json', 'r') as file: 
    data = json.load(file)

# Initialize a counter for vehicles
vehicle_image_count = defaultdict(int)

# Iterate over the data to count how many images belong to each vehicle
for entry in data.values():
    vehicle_id = entry['vehicleID']
    
    vehicle_image_count[vehicle_id] += 1
    
# Calculate the total number of vehicles and the total number of images
total_vehicles = len(vehicle_image_count)
total_images = sum(vehicle_image_count.values())

# Calculate the average number of images per vehicle
average_images_per_vehicle = total_images / total_vehicles if total_vehicles > 0 else 0


print(f"\nAverage number of images per vehicle: {average_images_per_vehicle:.2f}")

# Print the results
print("Total image counts per vehicle:")
for vehicle_id, count in vehicle_image_count.items():
    print(f"Vehicle ID {vehicle_id}: {count} images")

