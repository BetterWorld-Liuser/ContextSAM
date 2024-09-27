import json
import numpy as np
import cv2

json_path = "c:/Users/CPC_l/Desktop/ContextSAM实验记录/画出水塘/23.json"
with open(json_path) as f:
    data = f.read()

data = json.loads(data)

height = data["imageHeight"]
width = data["imageWidth"]
mask = np.zeros((height, width), dtype=np.uint8)

for shape in data["shapes"]:
    points = np.array(shape["points"], dtype=np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))

cv2.imwrite("mask.png", mask)
