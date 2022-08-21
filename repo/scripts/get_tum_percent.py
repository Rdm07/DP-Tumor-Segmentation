# Imports
import sys
import os
import cv2
import numpy as np

# Read image
image_path = sys.argv[1]
img = cv2.imread(image_path)

# Define colour lower limit values
r_lim = 15
g_lim = 0
b_lim = 240

# Opencv loads image in BGR format
# Count pixels for just tumour and whole tissue
img = cv2.imread(image_path)
img_h, img_w = img.shape[:2]
px_r = 0
px_tot = 0
for i in range(img_h):
	for j in range(img_w):
		if img[i][j][0] > b_lim:
			px_tot += 1
			if img[i][j][2] > r_lim:
				px_r += 1

# Calculate percent of tumour pixels out of tissue
tum_percent = np.round((px_r/px_tot),4)*100

# Write to file
o_filename = image_path.split(".")[0]+'.txt'
fid = open(o_filename, 'w')
fid.write('{} {}'.format("tumour percentage:", tum_percent))
fid.close()
