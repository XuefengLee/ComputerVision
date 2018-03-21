import cv2
import numpy as np
import otsu_threshold as otsu_threshold

BACKGROUND = 255

def connected_comp(image):
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
			
			# left
			if row - 1 >= 0 and elements[row-1,col] != 0:
				neighbours.append(elements[row-1,col])

			# right
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



def find_parent(parent,i):
    if parent[i] == -1:
        return i
    if parent[i]!= -1:
        return find_parent(parent,parent[i])

# A utility function to do union of two subsets
def union(parent,x,y):
    x_set = find_parent(parent, x)
    y_set = find_parent(parent, y)

    if x_set == y_set:
    	return

    if x_set > y_set:
    	parent[x_set] = y_set
    else:
		parent[y_set] = x_set




image = inputImage
#image =cv2.imread('ductile_iron2-0.jpg',0)
image = otsu_threshold.otsu(image)
image = connected_comp(image)
print image
# print image
# cv2.imshow('',image)
# cv2.waitKey(0)
