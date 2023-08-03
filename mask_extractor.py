import os
import cv2
import json
import numpy as np 
import csv

source_folder = os.path.join(os.getcwd(), "images")
json_path = "asp_jsons.json"                     # Relative to root directory
count = 0                                           # Count of total images saved
file_bbs = {
    }                                       # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 256				    # Dimensions should match those of ground truth image
MASK_HEIGHT = 256									

# Read JSON file
with open(json_path) as f:
  data = json.load(f)

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print("No BB. Skipping", key)
        return
    
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    file_bbs[key].append(all_points)
    return
for itr in data:
    file_name_json = data[itr]["filename"]
    file_bbs[file_name_json[:-4]] = []
    sub_count = 0               # Contains count of masks for a single ground truth image
    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            add_to_dict(data, itr, file_name_json[:-4], sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nDict size: ", len(file_bbs))
mask_folder = os.path.join("masks")
# make folders and copy image to new location
os.mkdir(mask_folder)

# For each entry in dictionary, generate mask and save in correponding 
# folder

f = open('meta_data.csv', 'w', newline='')
writer = csv.writer(f)
# write the header
header = ["image","mask"]
writer.writerow(header)
for itr in file_bbs:
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    try:
        for i in range(len(file_bbs[itr])):
            arr = np.array(file_bbs[itr][i])
            cv2.fillPoly(mask, [arr], color=(255))
        
        if(file_bbs[itr] != []):
            if("ROIs" in itr):
                cv2.imwrite(os.path.join(mask_folder, itr + "_mask.png") , mask)
                rows = [itr + ".png",itr + "_mask.png"]
            else:
                cv2.imwrite(os.path.join(mask_folder, itr + "_mask.jpg") , mask)
                rows = [itr + ".jpg",itr + "_mask.jpg"]
            writer.writerow(rows)
    except:
        print("Not found:", itr)
        continue
    count += 1
f.close()
print("Images saved:", count)