import cv2
import numpy as np
import argparse
import math


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input')
	parser.add_argument('--output')
	args = parser.parse_args()

	image = cv2.imread(args.input,0)
	image = cv2.medianBlur(image,9)
	threshold = otsu(image)
	image = apply_thresh(image, threshold)
	cv2.imwrite(args.output, image)

def otsu(image):	

	# convert the image to sorted 1-d array
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
