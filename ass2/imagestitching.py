from __future__ import print_function  #
import cv2
import numpy as np
import argparse
import os


def up_to_step_1(imgs):
	"""Complete pipeline up to step 3: Detecting features and descriptors"""

	imgs = detect(imgs)
	return imgs


def save_step_1(imgs, output_path='./output/step1'):
	"""Save the intermediate result from Step 1"""

	for filename,img,_,_ in imgs:

		cv2.imwrite(output_path + '/' + filename + '.jpg',img)


def up_to_step_2(imgs):
	"""Complete pipeline up to step 2: Calculate matching feature points"""
	
	data = detect(imgs)

	images = []
	length = len(data)
	for i in range(length):
		filename, img, kp, des = data[i]
		for j in range(i+1,length):
			images.append(matching(data[j], data[i]))


	return imgs, []


def save_step_2(imgs, match_list, output_path="./output/step2"):
	"""Save the intermediate result from Step 2"""
	# ... your code here ...
	pass

def up_to_step_3(imgs):
	imgs, data = detect(imgs)

	images = []
	length = len(data)
	for train in range(length):
		index,img,kp,des = data[train]
		for query in range(train+1,length):
			img, good = matching(data[query],data[train])
			images.append(img)
			H = ransac(good,data[query],data[train])

			Query = linear_transformation(data[query][1], H)

			cv2.imwrite('warps/' + str(train) + str(query) + '.jpg', Query)



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


def findH(src, dst):

	A = []
	for i in range(len(src)):
		x1, y1 = src[i][0], src[i][1]
		x2, y2 = dst[i][0], dst[i][1]
		A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
		A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

	U, S, Vh = np.linalg.svd(A)
	L = Vh[-1,:] / Vh[-1,-1]
	H = L.reshape(3, 3)

	return H

def detect(imgs):
	"""Complete pipeline up to step 3: Detecting features and descriptors"""

	data = []

	for filename,img in imgs:
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des = sift.detectAndCompute(gray,None)
		img = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		data.append((filename,img,kp,des))

	return data

def matching(query, train):

	query_index, query_img, query_kp, query_des = query
	train_index, train_img, train_kp, train_des = train


	# compute Euclidean distance matrix
	dists = np.sqrt(((query_des[:, :, None] - train_des[:, :, None].T) ** 2).sum(1))

	matches = []

	# get k nearest match point
	for i in range(dists.shape[0]):
		index = dists[i].argsort()[:2]

		# throw non-symmetry nearest relation
		index_2 = dists[:,index[0]].argsort()[0]
		if index_2 != i:
			continue

		matches.append((cv2.DMatch(index[0], i,dists[i][index[0]]),cv2.DMatch(index[1], i,dists[i][index[1]])))



	good = []
	for a,b in matches:
		if a.distance < 0.6 * b.distance:
			good.append(a)

	img2 = cv2.drawMatches(train_img,train_kp,query_img,query_kp,good,None)

	cv2.imwrite('matchings/' + str(train_index) + str(query_index) + '.jpg',img2)

	return img2,good

def ransac(good,query,train):

	src_pts = np.float32([ np.append(query[2][v.queryIdx].pt,1) for v in good ])
	dst_pts = np.float32([ train[2][v.trainIdx].pt for v in good ])

	min_num = np.inf
	Homo = None
	for _ in range(1000):
		idx = np.random.randint(len(src_pts), size=8)
		H = findH(src_pts[idx,:], dst_pts[idx,:])
		out = np.matmul(H,src_pts.transpose()).transpose()

		error = np.sqrt((out[:,:2]/out[:,[-1]] - dst_pts)**2)
		error = np.where(error > 3, 1, 0)
		num = np.sum(error)

		if num < min_num:
			min_num = num
			Homo = H

	return Homo


def linear_transformation(img, a):

	M, N, D = img.shape
	points = np.append(np.mgrid[0:N, 0:M].reshape((2, M*N)),[np.ones(M*N)],axis=0)

	affine_points = a.dot(points)
	affine_points /= affine_points[2, :]

	minX = np.min(affine_points[0, :])
	minY = np.min(affine_points[1, :])
	maxX = np.max(affine_points[0, :])
	maxY = np.max(affine_points[1, :])

	rangeW = int(maxX - minX)
	rangeH = int(maxY - minY)

	indY, indX = np.indices((rangeH, rangeW))
	newPictureIndex = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))
	newPictureIndex[0, :] += minX
	newPictureIndex[1, :] += minY

	mapToOriginalImgPoints = np.linalg.inv(a).dot(newPictureIndex)

	mapToOriginalImgPoints /= mapToOriginalImgPoints[2, :]

	dst = np.ones([rangeH, rangeW, D])

	map_x, map_y = mapToOriginalImgPoints[:2].reshape((2,rangeH,rangeW)).round().astype(np.int)

	dst = np.zeros([rangeH, rangeW, D])

	for i in range(rangeH):
		for j in range(rangeW):
			indexX = map_x[i][j]
			indexY = map_y[i][j]
			if indexX > N-1 or indexY > M-1 or indexX < 0 or indexY < 0:
				continue
			value = img[indexY][indexX]
			dst[i][j] = value

	return dst




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
	filelist = os.listdir(args.input)
	filelist.sort()
	for filename in filelist:
		print(filename)
		img = cv2.imread(os.path.join(args.input, filename))
		imgs.append((os.path.splitext(filename)[0],img))

	if args.step == 1:
		print("Running step 1")
		modified_imgs = up_to_step_1(imgs)
		save_step_1(modified_imgs, args.output)
	elif args.step == 2:
		print("Running step 2")
		modified_imgs = up_to_step_2(imgs)
		save_step_2(modified_imgs, args.output)
	elif args.step == 3:
		print("Running step 3")
		img_pairs = up_to_step_3(imgs)
		save_step_3(img_pairs, args.output)
	elif args.step == 4:
		print("Running step 4")
		panoramic_img = up_to_step_4(imgs)
		save_step_4(panoramic_img, args.output)

