import cv2
import numpy as np
import argparse
import math


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input')
	parser.add_argument('n')
	parser.add_argument('--output')
	args = parser.parse_args()

	image = cv2.imread(args.input,0)
	n = args.n
	image = grid_otsu(image, n)

	cv2.imwrite(args.output, image)

def grid_otsu(image,n):

	n = math.sqrt(int(n))
	img = np.zeros(image.shape)
	img_h, img_w = image.shape

	win_h = int(img_h/n) + int((img_h/n) > 0)
	win_w = int(img_w/n) + int((img_w/n) > 0)

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

if __name__ == "__main__":
	main()
