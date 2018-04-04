from __future__ import print_function  #
import cv2
import numpy as np
import argparse
import os


def up_to_step_1(imgs):
	"""Complete pipeline up to step 3: Detecting features and descriptors"""
	#output = set()
	for i,img in enumerate(imgs):
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des = sift.detectAndCompute(gray,None)
		img = cv2.drawKeypoints(gray,kp,img)
		#output.add(img)
		print(np.shape(des))
		#cv2.imwrite(str(i) + '.jpg',img)
	return imgs


def save_step_1(imgs, output_path='./output/step1'):
	"""Save the intermediate result from Step 1"""
	# ... your code here ...
	pass


def up_to_step_2(imgs):
	"""Complete pipeline up to step 2: Calculate matching feature points"""
	# ... your code here ...
	images = []
	for i,img in enumerate(imgs):
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des = sift.detectAndCompute(gray,None)
		img = cv2.drawKeypoints(gray,kp,img)
		images.append((img,kp,des))

	images = images[::-1]
	m,n = images[0],images[1]
	mdes, ndes = m[2], n[2]


	# compute Euclidean distance matrix
	dists = np.sqrt(((mdes[:, :, None] - ndes[:, :, None].T) ** 2).sum(1))
	matches = []

	# get k nearest match point
	for i in range(dists.shape[0]):
		index = dists[i].argsort()[:2]
		matches.append((cv2.DMatch(i, index[0], dists[i][index[0]])\
			,cv2.DMatch(i, index[1], dists[i][index[1]])))


	good = []
	for a,b in matches:

		if a.distance < 0.75*b.distance:
			# print(str(a.distance) + ' ' + str(b.distance))

			good.append(a)

	# img2 = cv2.drawMatchesKnn(m[0],m[1],n[0],n[1],good,None,flags=2)



	# Step 3
	# for m in good:
	# 	print(m.queryIdx)
	# 	print(m.trainIdx)
	src_pts = np.float32([ m[1][v.queryIdx].pt for v in good ])
	dst_pts = np.float32([ n[1][v.trainIdx].pt for v in good ])

	H = findHomography(src_pts[:8], dst_pts[:8])



	# cv2.imwrite('what1.jpg',img2)
	return imgs, []

def findHomography(src, dst):

	A = []
	for i in range(len(src)):
		x1, y1 = src[i][0], src[i][1]
		x2, y2 = dst[i][0], dst[i][1]
		A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
		A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

	# a = [-1,-1,-1,0,0,0,-1,-1,-1]
	# b = [0,0,0,-1,-1,-1,1,1,1]
	# mat = np.array([a,b])
	U, S, Vh = np.linalg.svd(A)
	L = Vh[-1,:] / Vh[-1,-1]
	H = L.reshape(3, 3)

	return H
	# print(np.matmul(H,[1,1,1]))
def save_step_2(imgs, match_list, output_path="./output/step2"):
	"""Save the intermediate result from Step 2"""
	# ... your code here ...
	pass


def up_to_step_3(imgs):


	findHomography(2,3)
	return imgs


def save_step_3(img_pairs, output_path="./output/step3"):
	"""Save the intermediate result from Step 3"""
	# ... your code here ...
	pass


def up_to_step_4(imgs):
	"""Complete the pipeline and generate a panoramic image"""
	# ... your code here ...
	return imgs[0]


def save_step_4(imgs, output_path="./output/step4"):
	"""Save the intermediate result from Step 4"""
	# ... your code here ...
	pass


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"step",
		help="compute image stitching pipeline up to this step",
		type=int
	)

	parser.add_argument(
		"input",
		help="a folder to read in the input images",
		type=str
	)

	parser.add_argument(
		"output",
		help="a folder to save the outputs",
		type=str
	)

	args = parser.parse_args()

	imgs = []
	for filename in os.listdir(args.input):
		print(filename)
		img = cv2.imread(os.path.join(args.input, filename))
		imgs.append(img)

	if args.step == 1:
		print("Running step 1")
		modified_imgs = up_to_step_1(imgs)
		save_step_1(imgs, args.output)
	elif args.step == 2:
		print("Running step 2")
		modified_imgs, match_list = up_to_step_2(imgs)
		save_step_2(modified_imgs, match_list, args.output)
	elif args.step == 3:
		print("Running step 3")
		img_pairs = up_to_step_3(imgs)
		save_step_3(img_pairs, args.output)
	elif args.step == 4:
		print("Running step 4")
		panoramic_img = up_to_step_4(imgs)
		save_step_4(img_pairs, args.output)
