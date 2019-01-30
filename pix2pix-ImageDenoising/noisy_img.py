import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

data_dir = "tensorflow-pix2pix-master\\document_dataset_original"
files = os.listdir(data_dir)


if __name__ =='__main__':
	for image in files:
		img = cv2.imread(data_dir + '/' + image)
		img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
		cv2.imwrite("tensorflow-pix2pix-master\\document_512x512_dataset\\"+image[:-4]+'.jpg',img)
		noisy = img + 100*np.random.randn(*img.shape)
	
		cv2.imwrite("tensorflow-pix2pix-master\\document_noisy512x512_dataset\\"+image[:-4]+'.png',noisy)
