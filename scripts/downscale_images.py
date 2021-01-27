import cv2
import os
from scipy import ndimage

lm_dirs = ["LM_Szenen_2_bis_6", "LM_Szenen_7_bis_11", "LM_Szenen_12_bis_16", "LM_Szenen_17_bis_21"]

for dir in lm_dirs:
    files = os.listdir(dir)
    for im_file in files:
        if im_file.endswith(".jpeg"):
            print(im_file)
            im_path = dir + "/" + im_file
            im_data = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

            scale_percent = 0.0741
            width = int(im_data.shape[1] * scale_percent)
            height = int(im_data.shape[0] * scale_percent)
            dim = (width, height)

            resized = cv2.resize(im_data, dim)
            if dim[0] > dim[1]:
                resized = ndimage.rotate(resized, 270)

            filename = "LM_Images_downscaled" + "/" + im_file
            cv2.imwrite(filename, resized)
