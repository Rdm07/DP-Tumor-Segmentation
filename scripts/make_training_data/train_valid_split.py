import os, sys
import random

from cv2 import validateDisparity

patches_folder = sys.argv[1]

svs_files = [name for name in os.listdir(patches_folder) if os.path.isdir(os.path.join(patches_folder, name))]

num_svs = len(svs_files)
valid_percent = 0.3
valid_ind = random.sample(range(num_svs), int(valid_percent*num_svs))

valid_text = ""
for i in valid_ind:
	valid_text = valid_text + svs_files[i] + " "

valid_text = valid_text + "\n"
tumor_data_list = [valid_text]

for i in range(len(svs_files)):
	if i not in valid_ind:
		tumor_data_list.append(svs_files[i]+"\n")

file1 = open(os.path.join(patches_folder, "tumor_data_list.txt"), 'w')
file1.writelines(tumor_data_list)
file1.close()