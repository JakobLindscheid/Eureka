import numpy as np

# Load the .npz file
data = np.load('/home/vandriel/Documents/GitHub/Eureka/eureka/outputs/eureka/2024-10-13_12-48-20/summary.npz')
# Check the contents (list of array names in the file)
print(data.files)

# Access each array by its name
for array_name in data.files:
    array_data = data[array_name]
    print(f"{array_name}: {array_data}")