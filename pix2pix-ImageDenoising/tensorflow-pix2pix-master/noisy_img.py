import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

data_dir = r'C:\Users\USER\Desktop\pix2pix-ImageDenoising\tensorflow-pix2pix-master\flower_dataset'
files = os.listdir(data_dir)
print("Outside main")
def noisy(data_dir,mean,var):
	mean = mean
	var = var
	for file in files:
		im = cv2.imread(data_dir + '/' + file)
		im = cv2.resize(im,(250,250),interpolation=cv2.INTER_AREA)
		
		cv2.imwrite("tensorflow-pix2pix-master\\rose_noisy_dataset\\"+data_dir[:-4]+'.png',noisy)


if __name__ =='__main__':
	#noisy(data_dir,0,0.2)
	for image in files:
		img = cv2.imread(data_dir + '/' + image)
		img = cv2.resize(img,(250,250),interpolation=cv2.INTER_AREA)
		cv2.imwrite("tensorflow-pix2pix-master\\rose_250x250_dataset\\"+image[:-4]+'.jpg',img)
		noisy = img + 30*np.random.randn(*img.shape)
		#noisy = np.clip(noisy, 0., 255.)
	
		cv2.imwrite("tensorflow-pix2pix-master\\rose_noisy_dataset\\"+image[:-4]+'.png',noisy)
	#print("done")
