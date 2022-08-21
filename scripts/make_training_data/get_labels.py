import os, sys
from traceback import print_tb
import cv2
import openslide

patches_folder = sys.argv[1]
svs_folder = sys.argv[2]
mask_folder = sys.argv[3]

for folder in os.listdir(patches_folder):
	if os.path.isdir(os.path.join(patches_folder, folder)):
		slide_name = folder
		slide_path = os.path.join(svs_folder, slide_name)
		oslide = openslide.OpenSlide(slide_path)
		slide_width = oslide.dimensions[0]
		slide_height = oslide.dimensions[1]
		print("SVS File: {}".format(slide_path))

		mask_name = folder.rsplit('.', 1)[0] + '.png'
		mask_path = os.path.join(mask_folder, mask_name)
		mask_img = cv2.imread(mask_path)
		print("PNG File: {}".format(mask_path))

		mask_width = mask_img.shape[1]
		mask_height = mask_img.shape[0]
		
		scale_h = mask_height/slide_height
		scale_w = mask_width/slide_width

		lab_file = os.path.join(patches_folder, folder, "label.txt")
		file1 = open(lab_file, 'w')

		for files in os.listdir(os.path.join(patches_folder, folder)):
			if '.txt' in files:
				continue
			else:
				file_name = files.split('.')[0]
				sc_x = int(float(file_name.split('_')[0])*scale_w)
				sc_y = int(float(file_name.split('_')[1])*scale_h)
				sc_pw_x = int(float(file_name.split('_')[2])*scale_w)
				sc_pw_y = int(float(file_name.split('_')[3])*scale_h)

				sub_mask = mask_img[sc_y:sc_y+sc_pw_y, sc_x:sc_x+sc_pw_x]

				white_px = 0
				tot_px = 0
				for x in range(1, sub_mask.shape[1]):
					for y in range(1, sub_mask.shape[0]):
						tot_px +=1
						if sub_mask[y][x][0] == 255:
							white_px += 1

				if white_px/tot_px > 0.1:
					s = "{} 1\n".format(files)
				else:
					s = "{} 0\n".format(files)
				file1.writelines(s)

	file1.close()
