import cv2
import numpy as np
import otsu_threshold as otsu_threshold
import argparse
BACKGROUND = 255



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input')
	parser.add_argument('--size')
	parser.add_argument('--optional_output')
	args = parser.parse_args()

	image = cv2.imread(args.input,0)
	n = int(args.size)

	image = connected_comp(image)
	n, deleted = count(image,n)
	print(n)

	if args.optional_output:
		image = convert_image(image, deleted)
		cv2.imwrite(args.optional_output, image)


def connected_comp(image, threshold=50):
	img_h, img_w = image.shape
	elements = np.zeros(image.shape,dtype=int)
	label = 1
	linked = {}
	parent = [-1]

	for row in range(img_h):
		for col in range(img_w):
			if image[row,col] == BACKGROUND:
				continue

			neighbours = []
			
			# up
			if row - 1 >= 0 and elements[row-1,col] != 0:
				neighbours.append(elements[row-1,col])

			# left
			if col - 1 >= 0 and elements[row,col-1] != 0:
				neighbours.append(elements[row,col-1])
			

			if len(neighbours) == 0:
				elements[row,col] = label
				linked[label] = label
				label += 1
				parent.append(-1)
			else:
				elements[row,col] = min([neighbour for neighbour in neighbours])
				if len(neighbours) > 1:
					union(parent,neighbours[0],neighbours[1])

	for row in range(img_h):
		for col in range(img_w):
			if image[row,col] == BACKGROUND:
				continue
			elements[row,col] = find_parent(parent,linked[elements[row,col]])

	return elements


def count(elements,threshold):

	num = 0
	row, col = elements.shape
	deleted = []
	nodule_dict = {}
	labels = np.zeros(elements.shape)
	for i in range(row):
		for j in range(col):
			if elements[i,j] not in nodule_dict:
				nodule_dict[elements[i,j]] = 1
			else:
				nodule_dict[elements[i,j]] += 1

	for key in list(nodule_dict.keys()):
		if nodule_dict[key] > threshold:
			num += 1
		else:
			deleted.append(key)

	# throw background color
	return num - 1, deleted




def find_parent(parent,i):
	if parent[i] == -1:
		return i
	if parent[i]!= -1:
		return find_parent(parent,parent[i])

def union(parent,x,y):
	x = find_parent(parent, x)
	y = find_parent(parent, y)

	if x == y:
		return

	if x > y:
		parent[x] = y
	else:
		parent[y] = x

def convert_image(image, deleted):
	row, col = image.shape
	for i in range(row):
		for j in range(col):
			if image[i][j] in deleted:
				image[i][j] = 0

	amax = np.max(image)

	labels = []

	# normalize the matrix so each label is in the 0-255
	image = image/float(amax) * 255

	# convert gray to rgb
	image = np.stack((image,)*3, -1)*3%256
	image[:,:,2] = image[:,:,2]*3%256
	image[:,:,1] = image[:,:,1]*7%256

	# convert backgound color
	image[image == 0] = BACKGROUND

	return np.array(image, dtype=np.uint8)

if __name__ == "__main__":
	main()

