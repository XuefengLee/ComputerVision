import cv2
import numpy as np

def grid_otsu(image):
	img = np.zeros(image.shape)
	img_h, img_w = image.shape
	win_h, win_w = 50, 50
	row, col = (0,0)

	while row < img_h:
		new_row = min(row+win_h, img_h)
		col = 0
		while col < img_w:
			new_col = min(col+win_w, img_w)
			block = image[row:new_row, col:new_col]
			threshold = otsu(block)
			img[row:new_row, col:new_col] = apply_thresh(block, threshold)
			col = new_col
		row = new_row




	return np.array(img, dtype=np.uint8)


def otsu(image):	

	sorted_pix = image.ravel()
	length = float(len(sorted_pix))
	candidates = []

	for i in range(255):
	    backs = sorted_pix > i
	    noduels = sorted_pix <= i
	    sum1 = np.sum(backs,dtype=np.float32)
	    sum2 = np.sum(noduels,dtype=np.float32)
	    if sum1 == 0 or sum2 == 0:
	        candidates.append(0)
	        continue
	    w1 = sum1/length
	    w2 = sum2/length
	    u1 = np.sum(backs*sorted_pix)/sum1
	    u2 = np.sum(noduels*sorted_pix)/sum2
	    result = w1*w2*(u1-u2)**2
	    candidates.append(result)
	threshold = np.argmax(candidates)

	return threshold


def apply_thresh(image, threshold):

	image = (image > threshold)*255
	
	return np.array(image, dtype=np.uint8)
# image =cv2.imread('ductile_iron2-0.jpg',0)
# image = grid_otsu(image)
# cv2.imshow('image',image)
# cv2.waitKey(0)